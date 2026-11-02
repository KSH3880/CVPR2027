from ksh_jointpush.ksh_live_control import read_json, select_case, write_json


def cases():
    return [dict(file='pair-a', category='2+2'), dict(file='four-a', category='4'),
            dict(file='pair-b', category='2+2')]


def test_next_selection_tracks_filename_when_new_category_arrives():
    current = cases()
    assert select_case(current, 'pair-a', 44)['file'] == 'pair-b'
    current.append(dict(file='split-a', category='1+3'))
    assert select_case(current, 'pair-a', 44)['file'] == 'pair-b'
    assert select_case(current, 'four-a', 44)['file'] == 'split-a'


def test_missing_category_is_not_exit_or_another_category():
    assert select_case(cases(), 'pair-a', 41) is None
    assert select_case(cases(), 'pair-a', 42)['file'] == 'pair-a'
    assert select_case(cases(), 'pair-a', 43)['file'] == 'four-a'


def test_atomic_protocol_preserves_large_sequence_and_ignores_incomplete_read(tmp_path):
    path = tmp_path/'request.json'
    sequence = 1793299328412345678
    write_json(path, dict(sequence=sequence, code=44))
    assert read_json(path)['sequence'] == sequence
    assert not list(tmp_path.glob('*.tmp.*'))
    path.write_text('{')
    assert read_json(path, {}) == {}
