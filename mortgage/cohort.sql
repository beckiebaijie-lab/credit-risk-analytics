CREATE VIEW mortgage_cohort AS
WITH unique_origination AS (
    SELECT *, count(*) OVER (PARTITION BY loan_id) AS id_count
    FROM origination
), eligible AS (
    SELECT * FROM unique_origination
    WHERE split IS NOT NULL AND trim(loan_id) <> '' AND id_count = 1
      AND original_upb IS NOT NULL
), monthly AS (
    SELECT e.loan_id, p.report_index, count(*) AS row_count,
           max(p.delinquency) AS delinquency,
           max(CASE WHEN trim(p.zero_balance) = '01' THEN 1 ELSE 0 END) AS prepaid
    FROM eligible e
    JOIN performance p ON p.loan_id = e.loan_id
      AND p.report_index BETWEEN e.first_index AND e.first_index + 11
      AND p.report_index <= {{PERFORMANCE_CUTOFF}}
    GROUP BY e.loan_id, p.report_index
), observations AS (
    SELECT m.loan_id, max(m.row_count) AS max_rows_per_month,
           max(CASE WHEN m.delinquency >= 3 THEN 1 ELSE 0 END) AS ever_90dpd,
           min(CASE WHEN m.prepaid = 1 THEN m.report_index END) AS prepaid_month
    FROM monthly m
    GROUP BY m.loan_id
), coverage AS (
    SELECT m.loan_id,
           sum(CASE WHEN m.delinquency BETWEEN 0 AND 2 AND
                        (o.prepaid_month IS NULL OR m.report_index < o.prepaid_month)
                    THEN 1 ELSE 0 END) AS clean_months
    FROM monthly m JOIN observations o ON o.loan_id = m.loan_id
    GROUP BY m.loan_id
)
SELECT e.loan_id, e.first_payment, e.split,
       e.fico, e.occupancy, e.cltv, e.dti, e.original_upb, e.ltv,
       e.property_type, e.purpose, e.term,
       CASE
           WHEN o.max_rows_per_month > 1 THEN 'duplicate_performance_month'
           WHEN o.ever_90dpd = 1 THEN 'event'
           WHEN o.prepaid_month IS NOT NULL
                AND c.clean_months = o.prepaid_month - e.first_index THEN 'prepaid_without_event'
           WHEN c.clean_months = 12 THEN 'complete_without_event'
           ELSE 'insufficient_history'
       END AS outcome_status
FROM eligible e
LEFT JOIN observations o ON o.loan_id = e.loan_id
LEFT JOIN coverage c ON c.loan_id = e.loan_id;
