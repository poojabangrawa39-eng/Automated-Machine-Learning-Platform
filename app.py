
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error

st.set_page_config(page_title="Automated ML Platform", layout="wide")

st.title("🤖 Automated End-to-End ML Platform")

uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = df.drop_duplicates()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])

    target_keywords = ["price", "salary", "charges", "cost", "fare", "income", "amount"]
    suggested_target = df.columns[-1]

    for col in df.columns:
        for word in target_keywords:
            if word in col.lower():
                suggested_target = col
                break

    target = st.selectbox(
        "What do you want to predict?",
        df.columns,
        index=list(df.columns).index(suggested_target)
    )

    if st.button("🚀 Train Model"):

        data = df.copy().dropna()

        encoders = {}
        original_options = {}

        for col in data.columns:
            if data[col].dtype == "object":
                original_options[col] = sorted(data[col].astype(str).unique())
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le

        X = data.drop(target, axis=1)
        y = data[target]

        if y.nunique() <= 10 and y.dtype != "float64":
            problem_type = "Classification"
            model = RandomForestClassifier(random_state=42)
        else:
            problem_type = "Regression"
            model = RandomForestRegressor(random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.model = model
        st.session_state.X = X
        st.session_state.target = target
        st.session_state.problem_type = problem_type
        st.session_state.encoders = encoders
        st.session_state.original_options = original_options

        st.success("✅ Model Trained Successfully!")

        if problem_type == "Regression":
            st.write("R² Score:", round(r2_score(y_test, y_pred), 2))
            st.write("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
        else:
            st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 2))

if "model" in st.session_state:

    st.subheader("🔮 Predict New Result")

    input_values = []

    for col in st.session_state.X.columns:

        if col in st.session_state.original_options:
            selected = st.selectbox(f"Select {col}", st.session_state.original_options[col])
            encoded_value = st.session_state.encoders[col].transform([selected])[0]
            input_values.append(encoded_value)

        else:
            if (
                "age" in col.lower()
                or "children" in col.lower()
                or "room" in col.lower()
                or "bedroom" in col.lower()
                or "bathroom" in col.lower()
                or "bhk" in col.lower()
            ):
                value = st.number_input(
                    f"Enter {col}",
                    min_value=0,
                    value=int(round(st.session_state.X[col].mean())),
                    step=1
                )
            else:
                value = st.number_input(
                    f"Enter {col}",
                    value=float(st.session_state.X[col].mean())
                )

            input_values.append(value)

    if st.button("Predict"):
        prediction = st.session_state.model.predict([input_values])

        if st.session_state.problem_type == "Regression":
            st.success(f"Predicted {st.session_state.target}: {round(prediction[0], 2)}")
        else:
            st.success(f"Predicted {st.session_state.target}: {prediction[0]}")