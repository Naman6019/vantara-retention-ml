from typing import List, Optional

from pydantic import BaseModel, Field


class CustomerFeaturesInput(BaseModel):
    customer_id: Optional[float] = Field(default=None, description="Unique customer ID")
    frequency: int = Field(..., ge=0, description="Total number of distinct historical purchases")
    total_items_bought: int = Field(..., ge=0, description="Total units/items purchased")
    unique_skus_bought: int = Field(..., ge=0, description="Number of unique products bought")
    monetary_total: float = Field(..., description="Net total historical spend (£)")
    recency: float = Field(..., ge=0, description="Days since last purchase")
    customer_age_days: float = Field(..., ge=0, description="Days since first purchase")
    monetary_avg: float = Field(..., description="Average spend per order (£)")
    avg_basket_size: float = Field(..., ge=0, description="Average items per basket")
    interpurchase_interval_mean: float = Field(default=0.0, ge=0)
    interpurchase_interval_std: float = Field(default=0.0, ge=0)
    purchase_frequency_trend: float = Field(default=0.0, description="Trend slope between -1 and 1")
    return_rate: float = Field(default=0.0, ge=0, le=1.0, description="Return rate ratio [0, 1]")
    cancellation_rate: float = Field(default=0.0, ge=0, le=1.0, description="Cancellation rate [0, 1]")
    seasonal_concentration: float = Field(default=0.0, ge=0, le=1.0, description="Q4 spend ratio [0, 1]")
    discount_sensitivity: float = Field(default=0.0, ge=0, le=1.0, description="Discounted items ratio [0, 1]")
    is_uk_customer: int = Field(default=1, ge=0, le=1, description="1 if UK customer, 0 otherwise")
    engagement_score: float = Field(default=50.0, ge=0, le=100.0, description="Engagement composite [0, 100]")
    spend_last_90d: float = Field(default=0.0, ge=0, description="Net spend in the 90 days before the cutoff (value model baseline)")


class DriverImpact(BaseModel):
    feature: str
    value: float
    shap_impact: float
    direction: str


class PredictionResponse(BaseModel):
    customer_id: Optional[float]
    churn_probability: float = Field(..., description="Calibrated probability of churn [0, 1]")
    is_churn_risk: bool = Field(..., description="True if calibrated churn probability >= model-card threshold")
    risk_tier: str = Field(..., description="Low, Moderate, or High (from calibrated probability quantiles)")
    predicted_future_clv: float = Field(..., description="Predicted spend in the next 90 days (£)")
    value_at_risk: float = Field(..., description="calibrated P(churn) x predicted 90-day value (£)")
    predicted_segment: str = Field(..., description="Assigned business persona")
    recommended_action: str = Field(..., description="Retention action from the rule engine")
    recommendation_channel: str
    recommendation_reasons: List[str]
    narrative_explanation: str
    top_risk_drivers: List[DriverImpact]


class ModelMetadataResponse(BaseModel):
    platform: str
    version: str
    churn_model: str
    clv_model: str
    segmentation_model: str
    test_roc_auc: float
    test_clv_r2: float
    decision_threshold: float
    test_recall: float
    features_expected: List[str]
