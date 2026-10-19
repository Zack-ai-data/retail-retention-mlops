-- Customer segments produced by the RFM clustering model (src/model.py).
-- Replaced on every training run; the model version that produced them is recorded.
-- Not part of the transform step because it is model output, not derived directly from raw data.
CREATE TABLE IF NOT EXISTS marts.customer_segments (
    snapshot_date  date    NOT NULL,
    customer_id    integer NOT NULL,
    segment_id     integer NOT NULL,
    segment_name   text    NOT NULL,
    model_version  text    NOT NULL,
    PRIMARY KEY (snapshot_date, customer_id)
);
