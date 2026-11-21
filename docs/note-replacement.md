# Note replacement

Repair a damaged passage by working out which notes were played and playing them back with clean recordings of the same piano. Nothing is guessed, so timing and notes stay as performed. This replaced blind generative inpainting as the plan after the first trial (see `inpainting-research.md`).

## Steps

1. **Transcribe.** `pip install transkun "setuptools<81"`, then `transkun song.wav song.mid --device cpu` (about 70 s for a 4 minute song on CPU). On Child of Light: 680 notes, 99% with a matching attack and pitch peak in the audio, and Daniel judged it note-perfect by ear.
2. **Build a note bank** from sample recordings: `NoteBank.from_files([...])` in `vibesound/notebank.py`. Each file is one pass of single notes played one after another. Pitch, loudness and note boundaries are detected automatically; "pedal" in the file name marks a sustain-pedal pass.
3. **Replace.** `replace_region(audio, sr, notes, bank, start, end)` renders the transcribed notes from the bank (including notes still ringing from before the region), matches level to the original in 200 Hz to 4 kHz, and crossfades just outside the region.

## Recording the samples

Felt on, usual mics, positions and gain; raw unprocessed WAV; 30 s of room silence first. Every key low to high, each note left to ring out. Passes in order of importance: soft, medium, soft with pedal down, medium with pedal down, firm. Extra touches (staccato) are optional.

## Not done yet

- MCP tools (`build_note_bank`, `replace_notes`) and reading MIDI plus pedal events into `Note` lists.
- Sympathetic resonance between notes under pedal is only as good as the pedal-down samples.
- Velocity to loudness mapping is a linear guess (`velocity_to_level`); calibrate against real samples.
- Tested only on synthetic notes so far.
