WITH eligible AS (
    SELECT loan_id, first_payment, original_upb,
           CAST(outcome_status = 'event' AS INTEGER) AS event
    FROM mortgage_cohort
    WHERE split = 'test'
      AND outcome_status IN ('event', 'complete_without_event', 'prepaid_without_event')
), decisions AS (
    SELECT settings.model, settings.calibration_target, loans.first_payment,
           loans.original_upb, loans.event,
           CAST(scores.probability < settings.threshold AS INTEGER) AS kept
    FROM eligible AS loans
    JOIN audit_scores AS scores ON scores.loan_id = loans.loan_id
    JOIN audit_thresholds AS settings ON settings.model = scores.model
), counts AS (
    SELECT model, calibration_target,
           substr(first_payment, 1, 4) || '-' || substr(first_payment, 5, 2) AS period,
           count(*) AS loans, sum(event) AS events, sum(kept) AS loans_retained,
           sum(event * kept) AS events_retained, sum(original_upb * kept) AS retained_balance
    FROM decisions
    GROUP BY model, calibration_target, first_payment

    UNION ALL

    SELECT model, calibration_target, 'Portfolio',
           count(*), sum(event), sum(kept), sum(event * kept), sum(original_upb * kept)
    FROM decisions
    GROUP BY model, calibration_target
)
SELECT model, calibration_target, period, loans, events, loans_retained, events_retained,
       events - events_retained AS events_flagged, retained_balance,
       1.0 * events / loans AS observed_rate,
       1.0 * loans_retained / loans AS retention,
       1.0 * events_retained / nullif(loans_retained, 0) AS retained_event_rate,
       1.0 * (events - events_retained) / nullif(events, 0) AS capture_rate
FROM counts
ORDER BY model, calibration_target, period;
