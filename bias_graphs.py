import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Load the JSON data
with open("gender_bias_analysis_report_20241120_180709.json", "r") as file:
    data = json.load(file)

# Extract necessary data
model_variants = data["model_variant_analysis"]
categories = data["bias_analysis"]["category_analysis"]

# Prepare data for graphs
model_names = list(model_variants.keys())
overall_scores = [model["overall_bias_score"] for model in model_variants.values()]
male_scores = [model["by_gender"]["male"] for model in model_variants.values()]
female_scores = [model["by_gender"]["female"] for model in model_variants.values()]

# Category analysis
category_names = list(categories.keys())
male_category_scores = [categories[cat]["male_bias"] for cat in category_names]
female_category_scores = [categories[cat]["female_bias"] for cat in category_names]

# Create the "graphs" directory if it doesn't exist
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

# 1. Bar Chart: Overall bias scores by model
plt.figure(figsize=(8, 5))
plt.bar(model_names, overall_scores, color="skyblue")
plt.title("Overall Bias Scores by Model")
plt.ylabel("Bias Score")
plt.xlabel("Model Variant")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "overall_bias_scores_by_model.png"))

# 2. Stacked Bar Chart: Male and Female bias scores by model
x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, male_scores, width, label="Male", color="lightcoral")
plt.bar(x + width / 2, female_scores, width, label="Female", color="lightgreen")
plt.title("Bias Scores by Gender and Model")
plt.ylabel("Bias Score")
plt.xlabel("Model Variant")
plt.xticks(x, model_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bias_scores_by_gender_and_model.png"))

# 3. Grouped Bar Chart: Male and Female bias scores by category
x = np.arange(len(category_names))

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, male_category_scores, width, label="Male", color="gold")
plt.bar(x + width / 2, female_category_scores, width, label="Female", color="teal")
plt.title("Bias Scores by Category and Gender")
plt.ylabel("Bias Score")
plt.xlabel("Category")
plt.xticks(x, category_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bias_scores_by_category_and_gender.png"))

# 4. Line Graph: Mean bias scores by category
plt.figure(figsize=(8, 5))
plt.plot(
    category_names, male_category_scores, marker="o", label="Male Bias", color="blue"
)
plt.plot(
    category_names, female_category_scores, marker="o", label="Female Bias", color="red"
)
plt.title("Mean Bias Scores by Category")
plt.ylabel("Bias Score")
plt.xlabel("Category")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mean_bias_scores_by_category.png"))

# 5. Heatmap: Correlation of bias scores by gender and model
heatmap_data = np.array([male_scores, female_scores]).T  # Transpose for heatmap
sns.heatmap(
    heatmap_data,
    annot=True,
    xticklabels=["Male", "Female"],
    yticklabels=model_names,
    cmap="coolwarm",
    fmt=".2f",
)
plt.title("Heatmap of Bias Scores by Gender and Model")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmap_bias_scores.png"))

print("Graphs generated and saved successfully!")
