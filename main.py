import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    try:
        # -----------------------------
        # Create output folder if needed
        # -----------------------------
        os.makedirs("output", exist_ok=True)

        # -----------------------------
        # Load dataset
        # -----------------------------
        file_path = "dataset/sample_data.csv"
        df = pd.read_csv(file_path)

        print("\nOriginal Dataset:\n")
        print(df)

        # -----------------------------
        # Dataset Information
        # -----------------------------
        print("\nDataset Information:\n")
        print(df.info())

        print("\nMissing Values:\n")
        print(df.isnull().sum())

        # -----------------------------
        # Remove Duplicate Rows
        # -----------------------------
        df.drop_duplicates(inplace=True)

        # -----------------------------
        # Fill Missing Numerical Values
        # -----------------------------
        df["Age"] = df["Age"].fillna(df["Age"].mean())
        df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

        # -----------------------------
        # Fill Missing Categorical Values
        # -----------------------------
        df["Department"] = df["Department"].fillna(
            df["Department"].mode()[0]
        )

        # -----------------------------
        # Encode Categorical Columns
        # -----------------------------
        encoder = LabelEncoder()

        df["Department"] = encoder.fit_transform(df["Department"])

        # -----------------------------
        # Standardize Numerical Columns
        # -----------------------------
        scaler = StandardScaler()

        df[["Age", "Salary"]] = scaler.fit_transform(
            df[["Age", "Salary"]]
        )

        # -----------------------------
        # Boxplot Before Removing Outliers
        # -----------------------------
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df[["Age", "Salary"]])
        plt.title("Boxplot Before Removing Outliers")
        plt.savefig("output/boxplot.png")
        plt.close()

        # -----------------------------
        # Remove Outliers using IQR
        # -----------------------------
        Q1 = df[["Age", "Salary"]].quantile(0.25)
        Q3 = df[["Age", "Salary"]].quantile(0.75)

        IQR = Q3 - Q1

        df = df[
            ~(
                (
                    (df[["Age", "Salary"]] < (Q1 - 1.5 * IQR))
                    | (df[["Age", "Salary"]] > (Q3 + 1.5 * IQR))
                ).any(axis=1)
            )
        ]

        # -----------------------------
        # Save Cleaned Dataset
        # -----------------------------
        output_file = "output/cleaned_data.csv"
        df.to_csv(output_file, index=False)

        print("\nData Cleaning Completed Successfully.")

        print("\nCleaned Dataset:\n")
        print(df)

        print(f"\nCleaned dataset saved to: {output_file}")
        print("Boxplot saved in output folder.")

    except FileNotFoundError:
        print("Error: Dataset file not found.")

    except pd.errors.EmptyDataError:
        print("Error: Dataset is empty.")

    except PermissionError:
        print("Permission denied while accessing files.")

    except Exception as error:
        print(f"Unexpected Error: {error}")


if __name__ == "__main__":
    main()