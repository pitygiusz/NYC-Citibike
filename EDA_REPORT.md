# NYC Parking Violations Analysis - EDA Report

This report summarizes the findings from the Exploratory Data Analysis (EDA) of the NYC Parking Violations dataset for the fiscal year 2015.

## 1. Parking Violations by Registration State

**Insight:**
- Almost 78% of all violations were made by cars registered in **New York**, which is expected as we are analyzing New York City.
- Approximately one in ten parking tickets were given to cars registered in neighboring **New Jersey**.
- 2.5% of violations were made by cars from **Pennsylvania**, which does not directly border New York City.

![Top 10 States](plots/top_10_states.png)

## 2. Top Violation Codes

**Insight:**
- Violation Code 21 (NO PARKING – STREET CLEANING, fine $45) is the most frequent, accounting for 13.88% of violations, followed closely by Code 38 (FAIL TO DISPLAY MUNI METER RECEIPT, fine $35) at 12.08%.
- The top three violation codes (21, 38, and 14) together make up over 34% of all violations, highlighting their prominence.
- Despite the dominance of these top codes, the overall distribution of violations is fairly spread out, showing that issues occur across many categories rather than being concentrated in a single area.

![Top 10 Violation Codes](plots/top_10_violation_codes.png)

## 3. Temporal Patterns

### Day of the Week
**Insight:**
- Tuesday, Thursday, and Friday have the highest numbers of violations, with Wednesday and Monday coming a bit behind.
- Sunday has by far the lowest number of violations, showing a sharp decrease compared to all other days.
- The distribution of violations is uneven, with significantly higher counts on weekdays and a clear drop during the weekend, suggesting that violation frequency is closely tied to weekday operational or activity levels.

![Violations by Day](plots/violations_by_day.png)

### Month
**Insight:**
- The fiscal year runs from the beginning of July to the end of June.
- January and June had the highest number of violations in Fiscal Year 2015 (around 11%), likely because January marks the start of the calendar year (“fresh start”) and June is the last month of the fiscal year (catching up with targets).
- Most other months (March–October) show average percentages, indicating that violations are fairly evenly spread throughout the fiscal year.
- February and December had the lowest percentages, likely due to having the fewest working days.

![Violations by Month](plots/violations_by_month.png)

## 4. Vehicle Characteristics

### Top Vehicle Makes
**Insight:**
- **Ford** is the most frequently cited vehicle make (~13%), followed by **Toyota** (~10%) and **Honda** (~9%).
- This likely reflects the popularity of these brands in the NYC area rather than a specific tendency of these drivers to commit violations.

![Top 10 Vehicle Makes](plots/top_10_vehicle_makes.png)

## 5. Location Analysis

### Top Streets
**Insight:**
- **Broadway** has the highest number of violations, which is consistent with it being a major and long thoroughfare in NYC.
- Other major avenues like **3rd Ave**, **5th Ave**, and **Madison Ave** also feature prominently in the top 10.

![Top 10 Streets](plots/top_10_streets.png)

## 6. NY vs. Non-NY Registered Vehicles

**Insight:**
- Violation 71 (INSP. STICKER – EXPIRED/MISSING, fine $65) is highly specific to NY, accounting for 6.28% of violations there versus only 0.08% in Non-NY.
- Violation 36 (PHOTO SCHOOL ZONE SPEED VIOLATION, fine $50) is also more prevalent in NY (7.97%) compared to Non-NY (4.22%).
- Violation 14 (NO STANDING – DAY/TIME LIMITS, fine $115) shows the opposite trend, being more common in Non-NY (11.54%) than in NY (7.53%).
- Violation 21 (NO PARKING – STREET CLEANING, fine $45) is slightly higher in Non-NY (15.79%) than in NY (13.34%), but it remains one of the most frequent violations in both regions.

**Null hypothesis** (H₀): Violation codes are independent of whether a vehicle is registered in NY or not.

**Alternative hypothesis** (H₁): Violation codes are dependent on whether a vehicle is registered in NY or not (i.e., there is an association).

The chi-square test returned p = 0.00, which is below any conventional significance level (e.g., 0.05). Therefore, **we reject the null hypothesis**.

**Interpretation**: There is a significant association between violation codes and NY registration status. This means that the types of violations issued differ between NY-registered and Non-NY-registered vehicles. For example, some violations (like 71 – INSP. STICKER EXPIRED/MISSING) occur almost exclusively in NY, while others are more common outside NY.

![NY vs Non-NY Violations](plots/ny_vs_non_ny.png)

## 7. Vehicle Age Analysis

**Null hypothesis** (H₀): Violation codes are independent of vehicle age group.

**Alternative hypothesis** (H₁): Violation codes are dependent on vehicle age group (i.e., there is an association).

The chi-square test returned p = 0.00, which is below any typical significance level (e.g., 0.05). Therefore, we **reject the null hypothesis.**

**Interpretation:** There is a significant association between vehicle age group and violation codes. This means that certain types of violations are more common among vehicles of specific age groups, suggesting that vehicle age may influence the likelihood or type of violation issued.

## Conclusion

The analysis reveals clear patterns in parking violations in NYC. Enforcement is heavily concentrated on weekdays and specific violation types like street cleaning and meter receipts. There are distinct differences in violation behaviors (or enforcement focus) depending on whether a vehicle is registered in NY or elsewhere, or its age.
