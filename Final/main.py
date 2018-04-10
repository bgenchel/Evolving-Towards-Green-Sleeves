import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op 
import pdb
import pypianoroll
from datetime import datetime
from reverse_pianoroll import piano_roll_to_pretty_midi

NUM_ITERATIONS = 300
NUM_CANDIDATES = 150
SELECTION_SIZE = 40
NUM_MUTATIONS = 2
NOTE_STDDEV = 8 # just a range of notes around the median


GREEN_SLEEVES = {"notes": np.array([57, 60, 62, 64, 65, 64, 62, 59, 55, 
                           57, 59, 60, 57, 57, 56, 57, 59, 55, 52]),
                 "durations": np.array([2, 4, 2, 3, 1, 2, 4, 2, 3, 1, 
                                        2, 4, 2, 3, 1, 2, 4, 2, 2])}
SEQLEN = len(GREEN_SLEEVES["notes"])

def convert_to_piano_roll_mat(notes, durations):
    scaled_durations = durations*12
    onsets = np.array([np.sum(scaled_durations[:i]) for i in range(len(scaled_durations))])
    total_dur = np.sum(scaled_durations)
    output_mat = np.zeros([128, int(total_dur)])
    # pdb.set_trace()
    for i in range(len(notes) - 1):
        output_mat[int(notes[i]), int(onsets[i]):int(onsets[i+1])] = 1.0
    # pdb.set_trace()
    output_mat[int(notes[-1]), int(onsets[-1]):] = 1.0
    return output_mat

def array_from_function(function, shape):
    # pdb.set_trace()
    outarray = np.zeros(shape)
    for index in np.ndindex(outarray.shape):
        outarray[index] = function(index)
    return outarray

def compute_loss(cand_notes, cand_durs, truth_notes, truth_durs):
    note_mse = ((cand_notes - truth_notes)**2).mean(axis=None)
    dur_mse = ((cand_durs - truth_durs)**2).mean(axis=None)
    len_diff_loss = (np.sum(truth_durs) - np.sum(cand_durs))**2
    return note_mse + dur_mse + len_diff_loss

def init_candidates(num_candidates, series_len, mean=0, stddev=1):
    num_entries = num_candidates*series_len
    means = np.array([mean]*num_entries).reshape(num_candidates, series_len)
    stddevs = np.array([stddev]*num_entries).reshape(num_candidates, series_len)
    candidates = np.round(np.random.normal(means, stddevs))
    return candidates

def main(runpath):
    lowest_losses = [] # top loss in each generation
    avg_losses = [] # avg loss for each generation

    note_mean = np.round(np.mean(GREEN_SLEEVES['notes']))
    note_candidates = init_candidates(NUM_CANDIDATES, SEQLEN, note_mean, NOTE_STDDEV)

    unique_durations = list(set(GREEN_SLEEVES["durations"]))
    dur_candidates = np.array([np.random.choice(unique_durations) 
                               for _ in range(NUM_CANDIDATES*SEQLEN)])
    dur_candidates = dur_candidates.reshape(NUM_CANDIDATES, SEQLEN)

    for i in range(NUM_ITERATIONS):
        print("Iteration %i" % i)

        losses = np.array([compute_loss(note_candidates[r, :], dur_candidates[r, :],
            GREEN_SLEEVES["notes"], GREEN_SLEEVES["durations"]) for r in
            range(NUM_CANDIDATES)])
        avg_losses.append(np.mean(losses))
        sort_indices = np.argsort(losses)
        lowest_losses.append(losses[sort_indices[0]])

        notes_select = note_candidates[sort_indices[:SELECTION_SIZE], :]
        durs_select = dur_candidates[sort_indices[:SELECTION_SIZE], :]
        if NUM_MUTATIONS > 0:
            mutant_mean = note_candidates.mean(axis=None)
            mutant_stddev = np.std(note_candidates, axis=None)
            mutant_notes = init_candidates(NUM_MUTATIONS, SEQLEN, mutant_mean,
                    mutant_stddev)
            notes_select = np.concatenate((notes_select, mutant_notes), axis=0)
        note_candidates = array_from_function(
                lambda rc: np.random.choice(notes_select[:, rc[1]]),
                [NUM_CANDIDATES, SEQLEN])

        if NUM_MUTATIONS > 0:
            mutant_durs = np.array([np.random.choice(unique_durations) 
                                    for _ in range(NUM_MUTATIONS*SEQLEN)])
            mutant_durs = mutant_durs.reshape(NUM_MUTATIONS, SEQLEN)
            durs_select = np.concatenate((durs_select, mutant_durs), axis=0)
        dur_candidates = array_from_function(
                lambda rc: np.random.choice(durs_select[:, rc[1]]),
                [NUM_CANDIDATES, SEQLEN])

        output_midi = np.zeros([128, 1])
        for j in range(2):
            pr = convert_to_piano_roll_mat(notes_select[j, :], durs_select[j, :])
            output_midi = np.concatenate((output_midi, pr), axis=1)
        pm = piano_roll_to_pretty_midi(output_midi)
        path = op.join(runpath, 'midi', 'round_%i.mid' % i)
        print('Writing output midi file %s ...' % path)
        pm.write(path)
    # pdb.set_trace()
    json.dump({'losses': avg_losses}, open(op.join(runpath, 'avg_losses.json'), 'w'), indent=4)
    json.dump({'losses': lowest_losses}, open(op.join(runpath, 'lowest_losses.json'), 'w'), indent=4)
    json.dump({'num_iterations': NUM_ITERATIONS,
               'num_candidates_per_iteration': NUM_CANDIDATES,
               'selection_size': SELECTION_SIZE,
               'num_mutations': NUM_MUTATIONS,
               'note_stddev': NOTE_STDDEV}, open(op.join(runpath, 'info.json'), 'w'),
               indent=4)
    # plt.plot(avg_losses)

if __name__ == '__main__':
    run_dirname = datetime.now().strftime('%Y-%m-%d_%H:%M')
    run_dirpath = op.join(op.dirname(op.abspath(__file__)), 'runs', run_dirname)
    midi_path = op.join(run_dirpath, 'midi')
    if not op.exists(midi_path):
        os.makedirs(midi_path)
    main(run_dirpath)
