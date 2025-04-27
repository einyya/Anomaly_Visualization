# Transformer Visualization

Animated analysis of training dynamics in the Anomaly Transformer model.

We visualize and animate the evolution of key and query projection weights throughout the training process, providing a deeper understanding of the Transformer's behavior in anomaly detection tasks.

---

## Setup

To extract data for plotting:

- Clone the [original Anomaly Transformer repository](https://github.com/thuml/Anomaly-Transformer).
- Replace the `solver.py` file with the version provided in this project.

To produce the visualizations:

- Run the `plots.py` script using the paths to the extracted data.

---

## Animated Example

![Layer vs Layer Delta - Query](layer%20VS%20layer%20Delta%20(query).gif)

---

## Project Report

For a detailed analysis of the training process and insights gained:
[Transformer Visualization Report (PDF)](transformer_visualization_report.pdf)

---

## Credits

Based on the paper:  
**"Anomaly Transformer: Time Series Anomaly Detection With Association Discrepancy"**  
by Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long.

---
