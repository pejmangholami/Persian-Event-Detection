# Understanding the Pearson Correlation Coefficient

This document explains the Pearson correlation coefficient, its formula, and how to interpret the results. This is the same correlation metric calculated in the `evaluation_table_generator.tsx` file to measure the relationship between Topic Recall and Total Entropy.

## The Formula

The Pearson correlation coefficient, denoted as `r`, is calculated with the following formula:

```
r = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² * Σ(y - ȳ)²]
```

### Explanation of Components:

-   **`r`**: The final correlation coefficient, which will be a value between -1 and +1.
-   **`Σ`**: The "summation" symbol, which means to add up all the values in a series.
-   **`x`**: Represents the individual values for the first variable (e.g., each `Topic Recall` score).
-   **`y`**: Represents the individual values for the second variable (e.g., each `Total Entropy` score).
-   **`x̄` (x-bar)**: The mean (average) of all the values of the `x` variable.
-   **`ȳ` (y-bar)**: The mean (average) of all the values of the `y` variable.

In simple terms, the formula calculates how much the two variables change together. It compares the distance of each data point from its mean for both variables and summarizes this relationship into a single number.

## How to Interpret the Result (`r`)

The value of `r` tells you two things about the relationship between the variables: its **direction** and its **strength**.

### 1. Direction of the Relationship (Positive or Negative)

-   **Positive `r` (closer to +1):** This indicates a **positive correlation**. When one variable increases, the other variable also tends to increase.
    -   *Example:* If `r` were `+0.8`, it would mean that parameter settings that lead to higher Topic Recall also lead to higher Total Entropy.
-   **Negative `r` (closer to -1):** This indicates a **negative (or inverse) correlation**. When one variable increases, the other variable tends to decrease.
    -   *Example:* In this project, a negative correlation is **desirable**. An `r` value of `-0.6` means that parameter settings that result in a high Topic Recall are associated with a low Total Entropy, which is the ideal outcome.
-   **`r` close to 0:** This indicates **no linear relationship** between the variables. They behave independently of one another.

### 2. Strength of the Relationship

The absolute value of `r` (how far it is from 0) indicates the strength of the linear relationship.

-   **`|r|` ≥ 0.7:**  A **strong** correlation. The data points are very close to forming a straight line. The relationship is very predictable.
-   **0.3 ≤ `|r|` < 0.7:** A **moderate** correlation. There is a clear relationship, but there is also some scatter or variability in the data.
-   **`|r|` < 0.3:** A **weak** correlation. There is a hint of a relationship, but it is not very clear or predictable.

### Correlation Goals in This Project

The synthetic data was generated to meet the following specific correlation goals between **Topic Recall** and **Total Entropy**:

-   **ParsBERT:** Moderate negative correlation (target `r` between -0.6 and -0.7).
-   **mBERT:** Moderate negative correlation, but weaker than ParsBERT (target `r` between -0.4 and -0.5).
-   **Statistical Model:** Weak negative correlation (target `r` between -0.2 and -0.3).
