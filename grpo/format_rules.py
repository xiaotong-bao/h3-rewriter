import re
keys=["integrated_multimodal_description","overall_soundscape","non_diegetic_music"]

def check_format(raw, task, duration_s):
    src = {"task": task, "duration_s": duration_s}
    positions = {k:list(re.finditer(r'(?m)^\s*'+k+r'\s*:',raw)) for k in keys}
    complete = all(positions[k] for k in keys)
    unique = complete and all(len(positions[k]) == 1 for k in keys)
    order = complete and [positions[k][0].start() for k in keys] == sorted(positions[k][0].start() for k in keys)
    prefix_valid = False
    boundary_present = False
    nonempty = False
    shots_valid = False
    cut_timestamps_valid = False
    if unique and order:
        marks = [positions[k][0] for k in keys]
        prefix = raw[:marks[0].start()].strip()
        boundary_present = bool(re.fullmatch(
            r'For the target video, at 0\.00 seconds into the target video, (?:Image 1|<Picture 1>) '
            r'\((?:from )?(?:\[Shot 1\]|Shot 1)\) is fully referenced\.', prefix))
        prefix_valid = boundary_present if src['task'] == 'i2va' else not prefix
        sections = [raw[m.end():marks[i+1].start() if i < 2 else len(raw)].strip() for i,m in enumerate(marks)]
        nonempty = all(sections)
        shot_marks = list(re.finditer(r'\[Shot\s+(\d+)\]', sections[0]))
        shot_nums = [int(m[1]) for m in shot_marks]
        shots_valid = bool(shot_nums) and shot_nums == list(range(1,len(shot_nums)+1))
        cut_timestamps_valid = all(re.match(r'\s*At\s+\d{2}:\d{2}\.\d{3}\b', sections[0][m.end():]) for m in shot_marks[1:])
    timestamps = [(int(m),int(s),int(ms)) for m,s,ms in re.findall(r'\bAt\s+(\d{2}):(\d{2})\.(\d{3})',raw)]
    seconds = [m*60+s+ms/1000 for m,s,ms in timestamps]
    time_valid = all(s<60 and m*60+s+ms/1000<=float(src['duration_s']) for m,s,ms in timestamps)
    time_order = seconds == sorted(seconds) and len(seconds) == len(set(seconds))
    open_dialogue = raw.count('<d>')
    closed_dialogue = raw.count('</d>')
    dialogues = list(re.finditer(r'<d>\[([^\]\n]+)\]([^<>]+)</d>',raw))
    dialogue_valid = open_dialogue == closed_dialogue == len(dialogues)
    # A speaker may be defined well before its dialogue. Do not impose an
    # arbitrary character window; actual attribution needs semantic review.
    speakers_present = all(re.search(r'\bS\d+\b',raw[:m.start()]) for m in dialogues)
    checks = {'three_required_fields':bool(complete),'unique_fields':bool(unique),'field_order':bool(order),
        'nonempty_fields':bool(nonempty),'legal_prefix_and_required_boundary':bool(prefix_valid),
        'consecutive_shot_numbers':bool(shots_valid),'cuts_have_timestamps':bool(cut_timestamps_valid),
        'timestamps_within_duration':bool(time_valid),'timestamps_increasing':bool(time_order),
        'no_think_tags':not bool(re.search(r'</?think\b',raw)), 'no_markdown_fences':'```' not in raw,
        'dialogue_tags_and_language':bool(dialogue_valid),'dialogue_speaker_label_present':bool(speakers_present)}
    return checks
