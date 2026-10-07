"""Publish explicitly selected checkpoint directories with SHA256 receipts."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

import boto3
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--folders', type=Path, required=True, help='JSON list of relative checkpoint paths')
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--profile', default='r2w')
    parser.add_argument('--endpoint', default='https://f25b0ac4c45a2442f62961145a64d158.r2.cloudflarestorage.com')
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    bucket = 'data-transfer-research'
    base = 'turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/'
    folders = json.loads(args.folders.read_text())
    files = []
    for name in folders:
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert (len(relative.parts) == 3 and relative.parts[1] in ('run', 'pilot')
                and (relative.name.startswith('checkpoint-') or relative.name == 'final_adapter'))
        folder = args.root / relative
        required = ['adapter_model.safetensors', 'adapter_config.json']
        if relative.name.startswith('checkpoint-'):
            required += ['optimizer.pt', 'scheduler.pt', 'trainer_state.json']
            required += [f'rng_state_{rank}.pth' for rank in range(8)]
        assert all((folder / item).is_file() for item in required), f'Incomplete checkpoint: {name}'
        destination = ('trl_sft_official_v2_20261006/sft/' + '/'.join(relative.parts[1:])
                       if relative.parts[0] == 'trl_sft_official_v2_20261006' else name)
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                assert not path.is_symlink(), f'Symlink rejected: {path}'
                files.append((path, base + destination + '/' + path.relative_to(folder).as_posix()))
    assert files and len({key for _, key in files}) == len(files)
    receipt = dict(bucket=bucket, endpoint=args.endpoint, folders=folders,
                   files=len(files), bytes=sum(p.stat().st_size for p, _ in files), uploaded=False, ready=False)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    if not args.upload:
        return
    client = boto3.Session(profile_name=args.profile).client('s3', endpoint_url=args.endpoint, region_name='auto')

    def publish(pair):
        path, key = pair
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        sha = digest.hexdigest()
        size = path.stat().st_size
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            if error.response['Error']['Code'] not in ('404', 'NoSuchKey', 'NotFound'):
                raise
            head = None
        reused = bool(head and head['ContentLength'] == size and head.get('Metadata', {}).get('sha256') == sha)
        if head and not reused:
            raise RuntimeError(f'Existing object differs; refusing overwrite: {key}')
        if not reused:
            client.upload_file(str(path), bucket, key, ExtraArgs={'Metadata': {'sha256': sha}},
                               Config=TransferConfig(max_concurrency=4))
        head = client.head_object(Bucket=bucket, Key=key)
        assert head['ContentLength'] == size and head.get('Metadata', {}).get('sha256') == sha
        return dict(key=key, bytes=size, sha256=sha, reused=reused, verification='remote_size_and_sha256_metadata')

    artifacts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, artifact in enumerate(pool.map(publish, files), 1):
            artifacts.append(artifact)
            print(f'VERIFIED {index}/{len(files)} {artifact["key"]}', flush=True)
    receipt.update(uploaded=True, ready=True, verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), artifacts=artifacts)
    receipt_key = base + 'checkpoint_publications/' + args.receipt.name
    receipt['receipt_uri'] = 's3://' + bucket + '/' + receipt_key
    body = json.dumps(receipt, indent=2).encode()
    client.put_object(Bucket=bucket, Key=receipt_key, Body=body, ContentType='application/json',
                      Metadata={'sha256': hashlib.sha256(body).hexdigest()})
    assert client.get_object(Bucket=bucket, Key=receipt_key)['Body'].read() == body
    args.receipt.write_bytes(body)
    print('READY', receipt['receipt_uri'], flush=True)


if __name__ == '__main__':
    main()
