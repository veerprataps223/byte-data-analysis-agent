"""
Dataset Generator for Data Analysis Agent Benchmarks
Generates realistic datasets with clean distributions, known statistical properties,
anomalies, and clear business dimensions.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

def generate_saas_churn(n=250):
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=int(d)) for d in np.random.randint(0, 500, n)]
    
    plan_tiers = np.random.choice(["Starter", "Professional", "Enterprise"], size=n, p=[0.45, 0.35, 0.20])
    mrr_map = {"Starter": 49.0, "Professional": 199.0, "Enterprise": 799.0}
    
    # Base MRR with minor jitter
    mrr = [round(mrr_map[p] + np.random.normal(0, mrr_map[p] * 0.05), 2) for p in plan_tiers]
    
    # Usage hours (Enterprise uses much more)
    usage = []
    for p in plan_tiers:
        if p == "Starter":
            u = max(1.0, np.random.normal(25.0, 10.0))
        elif p == "Professional":
            u = max(5.0, np.random.normal(90.0, 25.0))
        else:
            u = max(20.0, np.random.normal(240.0, 60.0))
        usage.append(round(u, 1))
        
    # Support tickets
    tickets = np.random.poisson(lam=2.5, size=n)
    
    # Inject a few extreme outliers in support tickets and usage
    tickets[12] = 22  # Anomaly
    tickets[45] = 19  # Anomaly
    usage[33] = 950.0 # Anomaly extreme usage
    usage[88] = 875.0 # Anomaly
    
    # Net Promoter Score (-10 to 10 or 1 to 10)
    nps = []
    for t in tickets:
        base_nps = 9 - int(t * 0.8) + np.random.randint(-1, 2)
        nps.append(int(np.clip(base_nps, 1, 10)))
        
    contract_length = np.random.choice([1, 12, 24], size=n, p=[0.55, 0.35, 0.10])
    regions = np.random.choice(["North America", "EMEA", "APAC", "LATAM"], size=n, p=[0.45, 0.30, 0.15, 0.10])
    
    # Churn probability higher if: contract=1, low NPS, high tickets, low usage
    churn_flags = []
    statuses = []
    for i in range(n):
        p_churn = 0.15
        if contract_length[i] == 1:
            p_churn += 0.20
        if nps[i] <= 4:
            p_churn += 0.35
        if tickets[i] >= 5:
            p_churn += 0.20
        if plan_tiers[i] == "Starter":
            p_churn += 0.10
        p_churn = min(0.92, max(0.02, p_churn))
        
        churned = 1 if np.random.rand() < p_churn else 0
        churn_flags.append(churned)
        statuses.append("Cancelled" if churned else ("Active" if np.random.rand() > 0.08 else "Paused"))
        
    feedback_pool = [
        "Love the clean analytics dashboard and fast SQL exports.",
        "Pricing is too expensive for a small startup on Starter plan.",
        "Frequent API timeout errors and slow customer support response.",
        "Onboarding was confusing, needs clearer video tutorials and guides.",
        "Crucial tool for our weekly growth review, highly recommend.",
        "Lack of SSO and audit logs prevents company-wide enterprise rollout.",
        "Cancelled subscription due to departmental budget cuts in Q3.",
        "Great performance and reliable metric alerts, dev team loves it.",
        "Missing automated HubSpot integration, had to build custom webhook.",
        "High churn risk due to lack of advanced ML cohort segmentation."
    ]
    notes = [np.random.choice(feedback_pool) for _ in range(n)]

    df = pd.DataFrame({
        "customer_id": [f"CUST-{1000 + i}" for i in range(n)],
        "signup_date": [d.strftime("%Y-%m-%d") for d in dates],
        "plan_tier": plan_tiers,
        "mrr": mrr,
        "monthly_usage_hours": usage,
        "support_tickets": tickets,
        "nps_score": nps,
        "contract_length_months": contract_length,
        "region": regions,
        "account_status": statuses,
        "is_churned": churn_flags,
        "customer_feedback_notes": notes
    })
    return df

def generate_ecommerce_sales(n=300):
    start_date = datetime(2023, 6, 1)
    dates = [start_date + timedelta(days=int(d)) for d in np.random.randint(0, 365, n)]
    
    categories = np.random.choice(["Electronics", "Home & Office", "Apparel", "Industrial", "Software"], size=n)
    unit_prices = {
        "Electronics": 320.0,
        "Home & Office": 85.0,
        "Apparel": 45.0,
        "Industrial": 750.0,
        "Software": 199.0
    }
    
    quantities = np.random.randint(1, 15, size=n)
    # Inject bulk anomaly
    quantities[15] = 120
    quantities[80] = 95
    
    discounts = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.25], size=n, p=[0.4, 0.25, 0.15, 0.1, 0.1])
    segments = np.random.choice(["Consumer", "Corporate", "Small Business"], size=n, p=[0.5, 0.3, 0.2])
    channels = np.random.choice(["Online Store", "Enterprise Sales", "Retail Partner"], size=n, p=[0.6, 0.25, 0.15])
    
    prices = [round(unit_prices[c] * (1.0 + np.random.normal(0, 0.08)), 2) for c in categories]
    sales = [round(q * p * (1.0 - d), 2) for q, p, d in zip(quantities, prices, discounts)]
    # Margin % varies by category
    margin_rates = {"Electronics": 0.22, "Home & Office": 0.35, "Apparel": 0.48, "Industrial": 0.28, "Software": 0.78}
    profits = [round(s * (margin_rates[c] - (d * 0.5)), 2) for s, c, d in zip(sales, categories, discounts)]
    
    df = pd.DataFrame({
        "order_id": [f"ORD-{50000 + i}" for i in range(n)],
        "order_date": [d.strftime("%Y-%m-%d") for d in dates],
        "category": categories,
        "customer_segment": segments,
        "sales_channel": channels,
        "quantity": quantities,
        "unit_price": prices,
        "discount_pct": discounts,
        "sales_amount": sales,
        "net_profit": profits
    })
    return df

def generate_employee_attrition(n=200):
    departments = np.random.choice(["Engineering", "Sales", "Marketing", "Customer Success", "Finance"], size=n)
    salary_ranges = {
        "Engineering": (85000, 160000),
        "Sales": (60000, 140000),
        "Marketing": (55000, 115000),
        "Customer Success": (50000, 95000),
        "Finance": (70000, 130000)
    }
    
    salaries = [int(np.random.uniform(salary_ranges[d][0], salary_ranges[d][1])) for d in departments]
    tenure = np.random.exponential(scale=3.5, size=n)
    tenure = [round(min(15.0, max(0.3, t)), 1) for t in tenure]
    
    satisfaction = np.random.randint(1, 6, size=n) # 1 to 5
    overtime = np.random.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
    remote_pct = np.random.choice([0, 25, 50, 75, 100], size=n)
    promoted = np.random.choice([0, 1], size=n, p=[0.82, 0.18])
    
    attrition = []
    for s, t, sat, ot, prom in zip(salaries, tenure, satisfaction, overtime, promoted):
        p_leave = 0.12
        if sat <= 2:
            p_leave += 0.35
        if ot == "Yes":
            p_leave += 0.20
        if prom == 0 and t > 4:
            p_leave += 0.15
        if s < 65000:
            p_leave += 0.10
        p_leave = min(0.9, max(0.02, p_leave))
        attrition.append(1 if np.random.rand() < p_leave else 0)
        
    df = pd.DataFrame({
        "employee_id": [f"EMP-{8000 + i}" for i in range(n)],
        "department": departments,
        "annual_salary": salaries,
        "years_at_company": tenure,
        "job_satisfaction_score": satisfaction,
        "overtime_work": overtime,
        "remote_work_pct": remote_pct,
        "promoted_last_3_years": promoted,
        "attrition_status": ["Left" if a == 1 else "Current" for a in attrition],
        "is_attrited": attrition
    })
    return df

if __name__ == "__main__":
    df_saas = generate_saas_churn()
    df_saas.to_csv("datasets/saas_churn.csv", index=False)
    print("Generated datasets/saas_churn.csv with shape:", df_saas.shape)
    
    df_ecom = generate_ecommerce_sales()
    df_ecom.to_csv("datasets/ecommerce_sales.csv", index=False)
    print("Generated datasets/ecommerce_sales.csv with shape:", df_ecom.shape)
    
    df_emp = generate_employee_attrition()
    df_emp.to_csv("datasets/employee_attrition.csv", index=False)
    print("Generated datasets/employee_attrition.csv with shape:", df_emp.shape)
