"""Measure policy stability across first-payment months in the holdout."""
import pandas as pd


def monthly_policy(frame, probabilities, threshold):
    """Apply a locked threshold and aggregate counts before calculating rates."""
    retained = probabilities < threshold
    monthly = frame[["first_payment", "event"]].assign(
        retained=retained,
        retained_event=frame.event.to_numpy() * retained,
    ).groupby("first_payment", sort=True).agg(
        loans=("event", "size"),
        events=("event", "sum"),
        loans_retained=("retained", "sum"),
        events_retained=("retained_event", "sum"),
    )
    monthly["events_flagged"] = monthly.events - monthly.events_retained
    monthly["observed_rate"] = monthly.events / monthly.loans
    monthly["retention"] = monthly.loans_retained / monthly.loans
    monthly["retained_event_rate"] = monthly.events_retained / monthly.loans_retained.where(
        monthly.loans_retained > 0
    )
    monthly["capture_rate"] = monthly.events_flagged / monthly.events.where(monthly.events > 0)
    monthly.index = pd.to_datetime(monthly.index, format="%Y%m").strftime("%Y-%m")
    return monthly.rename_axis("first_payment_month").reset_index()
