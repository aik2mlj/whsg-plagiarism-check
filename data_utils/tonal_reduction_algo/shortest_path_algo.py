import networkx as nx
import numpy as np


def compute_rhythm_type(onset, nbpm=4, nspb=4):
    output = np.zeros_like(onset).astype(np.int64)

    half = nspb * 2 if nbpm == 4 else nspb * 3
    quarter = nspb
    if nspb == 4:
        eighth = 2
    else:
        raise NotImplementedError

    is_half = onset % half == 0

    is_quarter = np.logical_and(onset % half != 0, onset % quarter == 0)
    is_eighth = np.logical_and(onset % quarter != 0, onset % eighth == 0)
    is_sixteenth = onset % eighth != 0

    output[is_half] = 0
    output[is_quarter] = 1
    output[is_eighth] = 2
    output[is_sixteenth] = 3
    return output


def return_rhythm_score(note_matrix, nbpm, nspb):
    rhythm_type = compute_rhythm_type(note_matrix[:, 0], nbpm, nspb)
    rhythm_coef = np.zeros_like(rhythm_type).astype(np.float32)
    rhythm_coef[rhythm_type == 0] = 1.
    rhythm_coef[rhythm_type == 1] = 1.
    rhythm_coef[rhythm_type == 2] = 1.0
    rhythm_coef[rhythm_type == 3] = 1.
    return rhythm_coef


def return_chord_tone_score(note_matrix):
    chord_tone_coef = np.zeros(len(note_matrix), dtype=np.float32)
    chord_tone_coef[note_matrix[:, -1] == 1] = 0.6
    chord_tone_coef[note_matrix[:, -1] == 0] = 1.4
    return chord_tone_coef


def return_pitch_score(note_matrix):
    pitches = note_matrix[:, 1]
    highest_pitch = pitches.max()
    lowest_pitch = pitches.min()
    mid_pitch = (highest_pitch + lowest_pitch) / 2
    # mean_pitch = pitches.mean()
    if highest_pitch != lowest_pitch:
        pitch_coef = np.sqrt((pitches - mid_pitch) ** 2) / (highest_pitch - lowest_pitch)
        pitch_coef = -(pitch_coef - 0.5) * 0.1 + 1
    else:
        pitch_coef = pitches / pitches
    return pitch_coef


def compute_duration_type(duration, nbpm=4, nspb=4):
    half = nspb * 2 if nbpm == 4 else nspb * 3
    quarter = nspb
    if nspb == 4:
        eighth = 2
    else:
        raise NotImplementedError

    output = np.zeros_like(duration).astype(np.int64)

    is_half = duration >= eighth
    is_quarter = np.logical_and(duration < half, duration >= quarter)
    is_eighth = np.logical_and(duration < quarter, duration >= eighth)
    is_sixteenth = duration < eighth

    output[is_half] = 0
    output[is_quarter] = 1
    output[is_eighth] = 2
    output[is_sixteenth] = 3
    return output


def return_duration_score(note_matrix, nbpm, nspb):
    duration_type = compute_duration_type(note_matrix[:, 2], nbpm, nspb)
    duration_coef = np.zeros_like(duration_type).astype(np.float32)
    duration_coef[duration_type == 0] = 0.9
    duration_coef[duration_type == 1] = 0.95
    duration_coef[duration_type == 2] = 1.05
    duration_coef[duration_type == 3] = 1.1
    return duration_coef


def detect_rel_type(note_matrix, i, j):
    bar_thresh = 2
    # can change to onset distance
    relation = 5
    bar_id1, bar_id2 = note_matrix[i, 3], note_matrix[j, 3]
    pitch1, pitch2 = note_matrix[i, 1], note_matrix[j, 1]
    chord_id1, chord_id2 = note_matrix[i, 6], note_matrix[j, 6]

    if (bar_id1 - bar_id2) < bar_thresh:
        if pitch1 == pitch2:
            relation = 0
        elif (pitch1 - pitch2) % 12 == 0:
            relation = 1

        elif 1 <= np.abs(pitch1 - pitch2) <= 2:
            relation = 2
        elif np.abs(pitch1 - pitch2) % 12 in [1, 2, 10, 11]:
            relation = 3

    if relation == 5 and chord_id1 == chord_id2:
        relation = 4

    return relation


def create_adj_matrix(note_matrix, nbpm, nspb, dist_param=1.6,
                      rhy_param=1., chord_param=1., pitch_param=1.,
                      dur_param=1.):
    n_note = len(note_matrix)
    adj_matrix = -np.ones((n_note, n_note))
    # rel_score_map = {0: 0.1, 1: 0.1, 2: 0.3, 3: 1, 4: 1.5, 5: 3}
    rel_score_map = {0: 0.1, 1: 1, 2: 0.3, 3: 1.3, 4: 1.5, 5: 3}

    rhythm_coef = return_rhythm_score(note_matrix, nbpm, nspb) ** rhy_param
    chord_tone_coef = return_chord_tone_score(note_matrix) ** chord_param
    pitch_coef = return_pitch_score(note_matrix) ** pitch_param
    dur_coef = return_duration_score(note_matrix, nbpm, nspb) ** dur_param

    for i in range(n_note):
        for j in range(i + 1, n_note):
            rel_type = detect_rel_type(note_matrix, i, j)
            rel_score = rel_score_map[rel_type]
            dist = (j - i) ** dist_param
            edge_weight = \
                (rel_score + dist) * rhythm_coef[j] * chord_tone_coef[j] * \
                pitch_coef[j] * dur_coef[j]
            adj_matrix[i, j] = edge_weight
    return adj_matrix


def compute_path_length(G, p):
    length = 0
    for i in range(1, len(p)):
        length += G.edges[p[i - 1], p[i]]['weight']
    return length


def find_tonal_shortest_paths(note_matrix, nbpm, nspb, num_path=1,
                              dist_param=1.6, rhy_param=1., chord_param=1.,
                              pitch_param=1., dur_param=1.):
    if note_matrix.shape[0] == 0:
        return [None for _ in range(num_path)]
    n_node = len(note_matrix)

    adj_mat = create_adj_matrix(note_matrix, nbpm, nspb,
                                dist_param, rhy_param, chord_param,
                                pitch_param, dur_param)

    G = nx.DiGraph()
    for i in range(len(note_matrix)):
        G.add_node(i, data=note_matrix[i])

    for i in range(len(G.nodes)):
        for j in range(i + 1, len(G.nodes)):
            G.add_edge(i, j, weight=adj_mat[i, j])

    all_paths = nx.shortest_simple_paths(G, source=0, target=n_node - 1,
                                         weight='weight')

    paths = []
    for path_id, path in enumerate(all_paths):
        paths.append({'path': path, 'distance': compute_path_length(G, path),
                      'reduction_rate': len(path) / n_node})
        if path_id == num_path - 1:
            break
    return paths
