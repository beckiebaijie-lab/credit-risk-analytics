CREATE TEMP VIEW checked AS
SELECT *,
       CASE
           WHEN id IS NULL OR trim(id) = '' THEN 'missing_id'
           WHEN count(*) OVER (PARTITION BY id) > 1 THEN 'duplicate_id'
           WHEN issue_month IS NULL OR substr(issue_month, 1, 4) NOT BETWEEN '2008' AND '2012' THEN 'invalid_issue_month'
           WHEN loan_status IS NULL OR loan_status NOT IN ('Fully Paid', 'Charged Off') THEN 'unknown_outcome'
           WHEN loan_amnt IS NULL OR loan_amnt <= 0 THEN 'invalid_loan_amount'
           WHEN term IS NULL OR trim(term) NOT IN ('36 months', '60 months') THEN 'invalid_term'
           ELSE 'eligible'
       END AS quality_status
FROM staged;

CREATE TEMP VIEW cohort AS
SELECT *, CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END AS charged_off
FROM checked
WHERE quality_status = 'eligible';
