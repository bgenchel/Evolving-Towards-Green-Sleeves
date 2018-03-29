(
s = Server.local;
s.boot;
)

(
SynthDef.new(\beep, {
	arg freq = 440, amp=0.3, dur=1;
	var sig1;
	//sig1 = SinOsc.ar(freq, 0, 0.3) * EnvGen.kr(Env.linen(0.05, 1, 0.1), doneAction:2);
	sig1 = SinOsc.ar(freq, 0, amp) * EnvGen.kr(Env.perc(0.01, 0.5), doneAction: 2);
	sig1 = sig1 + SinOsc.ar(2*freq, 0, 0.25*amp) * EnvGen.kr(Env.perc(0.01, dur), doneAction: 2);
	sig1 = sig1 + SinOsc.ar(3*freq, 0, 0.5*amp) * EnvGen.kr(Env.perc(0.01, dur), doneAction: 2);
	sig1 = sig1 + SinOsc.ar(4*freq, 0, 0.25*amp) * EnvGen.kr(Env.perc(0.01, dur), doneAction: 2);
	sig1 = FreeVerb.ar(sig1, 0.7, 0.5, 0.7);
	Out.ar(0, sig1);
	Out.ar(1, sig1);
}).add;
)

// the main function, going to need to figure out how to make multiple of these play at the same time.
// Probably save that for the final.
(
// var notes, durations;
~playSequence = {
	arg notes, durations, repeat=1, tempo=2;
	var noteStream, durStream;
	noteStream = Pseq(notes, repeat).asStream;
	durStream = Pseq(durations, repeat).asStream;
	TempoClock.default.tempo = tempo;
	Task({
		var delta;
		while {
			delta = durStream.next;
			delta.notNil;
		} {
			Synth(\beep, [freq: noteStream.next.midicps, sustain: delta]);
			delta.yield;
		}
	}, TempoClock.default).play();
};

/*notes = [60, 62, 64, 62, 60];
durations = [3, 1, 3, 1, 3];
~playSequence.value(notes, durations, 1, 4);*/
)

(
var mse;

mse = {
	arg seq1, seq2;
	var sq_errs = Array.new(seq1.size);
	// "entered mse".postln;
	(0..(seq1.size - 1)).do({|index|
		sq_errs.add((seq1[index] - seq2[index])**2);
	});
	mean(sq_errs);
};

~fitnessFunction = {
	arg cand_notes, cand_durs, truth_notes, truth_durs;
	var fitness, note_mse, dur_mse, dur_dist;
	// "entered fitness function".postln;
	note_mse = mse.value(cand_notes, truth_notes);
	dur_mse = mse.value(cand_durs, truth_durs);
	// "mse calculations are fine".postln;
	dur_dist = abs(sum(cand_durs) - sum(truth_durs))/sum(truth_durs);
	fitness = (note_mse + dur_mse + dur_dist);
	fitness;
};
)

(
~transpose = {
	arg array;
	var tArr;
	array.postln;
	tArr = Array.fill2D(array[0].size, array.size, {|r, c|
		array[c][r];});
	tArr;
};
)

(
var gs_notes, gs_durs, cand1_notes, cand1_durs, cand2_notes, cand2_durs, cand3_notes, cand3_durs, num_candidates, fitnesses, most_fit_notes, t_most_fit_notes, most_fit_durs, t_most_fit_durs, tempo = 16;

// this is our goal state, basically.
gs_notes = [57, 60, 62, 64, 65, 64, 62, 59, 55, 57, 59, 60, 57, 57, 56, 57, 59, 55, 52];
gs_durs = [2, 4, 2, 3, 1, 2, 4, 2, 3, 1, 2, 4, 2, 3, 1, 2, 4, 2, 2];

num_candidates = 40;
fitnesses = Array.new(num_candidates);
most_fit_notes = Array.new(5);
most_fit_durs = Array.new(5);

// generate the first generation of candidates
cand1_notes = Array.fill2D(num_candidates, gs_notes.size, {gauss(mean(gs_notes), 6).round(1)});
cand1_durs = Array.fill2D(num_candidates, gs_durs.size, {(1 + 4.rand.round(1))});
/*Task({
	"playing through first candidates.".postln;
	(0..(cand1_notes.size-1)).do({
		arg index;
		var delta = sum(cand1_durs[index]);
		~playSequence.value(cand1_notes[index], cand1_durs[index], 1, tempo);
		delta.wait;
	})
}).play;*/

// get their fitnesses
(0..(cand1_notes.size - 1)).do({
	arg index;
	fitnesses.add(~fitnessFunction.value(cand1_notes[index], cand1_durs[index], gs_notes, gs_durs));
});

(0..(fitnesses.size-1)).do({
	arg index;
	fitnesses[index] = [fitnesses[index], index];
});
fitnesses.sort({|aa, bb| aa[0] < bb[0];});

(0..(4)).do({
	arg index;
	most_fit_notes.add(cand1_notes[fitnesses[index][1]]);
	most_fit_durs.add(cand1_durs[fitnesses[index][1]]);
});

most_fit_notes.postln;
most_fit_durs.postln;

t_most_fit_notes = ~transpose.value(most_fit_notes);
t_most_fit_durs = ~transpose.value(most_fit_durs);

cand2_notes = Array.fill2D(num_candidates, gs_notes.size, { | r, c | t_most_fit_notes[c].choose;});
cand2_durs = Array.fill2D(num_candidates, gs_durs.size, { | r, c | t_most_fit_durs[c].choose;});
//TempoClock.default.sched(sum(sum(cand1_durs)) + 10, {
TempoClock.default.sched(1, {
	"playing through second candidates".postln;
	(0..(cand2_notes.size-1)).do({
		arg index;
		var delta = sum(cand1_durs[index]);
		~playSequence.value(cand2_notes[index], cand2_durs[index], 1, tempo);
		delta;
	})
}).play;
)

// testing area
(
var arr;
arr = Array.new(19);
arr.add(1);
arr.add(2);
arr.add(2.6);
arr.postln;
)

3**2.postln;
abs(-1).postln;

a = Array.new(4);
a.postln;
a[0] = 10;
a.postln;

a = Array.fill2D(6, 5, {4.rand;});
sum(sum(a));
a.reshape(a[0].size, a.size);
a[:][0];
a.size;
a[0].size;
b = Array2D.fromArray(a.size, a[0].size, a);
b.colAt(0);
a.colAt(0);
a.add(Array.fill(6, {10.rand;}));
a.size.postln;
a.sort( { |aa, bb| aa[4] > bb[4];});

a = [[1, 4.rand], [2, 4.rand], [3, 4.rand], [4, 4.rand]];
a.sort({|aa, bb| aa[1] < bb[1]});

[1, 4, 3, 7].choose;

a = [1, 2];

