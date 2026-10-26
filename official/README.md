`falcon_authors_pairs.csv`: per-pair R2 of single-session NDT2 and GRU decoders (trained on one session, scored
on every other) on the FALCON private test split. Derived from the `*_eval_ndt2.csv` / `*_eval_rnn.csv` tables in
the FALCON authors' public checkpoint folder
(https://drive.google.com/drive/folders/1ijvmPcbyjHlLEoWWSSZrNm1bBojGgrZG), with H1 session labels S0–S12
mapped to their (shifted) dates. These numbers are on the private split with no context/test split, so they are
reported as protocol B "official" and are not directly comparable with locally computed rows.
