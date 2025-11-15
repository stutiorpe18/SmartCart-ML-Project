import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------
# PAGE SETTINGS
# ----------------------------------------------------
st.set_page_config(
    page_title="SmartCart – ML Powered Shopping System",
    layout="wide"
)

st.title("🛒 SmartCart – ML Powered Shopping System")
st.write("Browse products → Add to Cart → Get Smart ML Recommendations")

# ----------------------------------------------------
# LOAD PRODUCTS
# ----------------------------------------------------
@st.cache_data
def load_products():
    df = pd.read_csv("products.csv")

    df.columns = ["product", "price", "image"]

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(0).astype(int)

    df["label"] = pd.cut(
        df["price"],
        bins=[0, 50, 150, 300, 5000],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return df

products_df = load_products()

# ----------------------------------------------------
# LOAD ML MODEL + ENCODER + SCALER
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("best_model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("encoder.joblib")
    return model, scaler, encoder

model, scaler, encoder = load_model()

# ----------------------------------------------------
# SESSION STATE (CART)
# ----------------------------------------------------
if "cart" not in st.session_state:
    st.session_state.cart = []

# ----------------------------------------------------
# ADD TO CART FUNCTION
# ----------------------------------------------------
def add_to_cart(item):
    st.session_state.cart.append(item)

# ----------------------------------------------------
# RECOMMENDATION FUNCTION
# ----------------------------------------------------
def get_recommendation(product_name):
    try:
        encoded_value = encoder.transform([product_name])[0]
    except:
        return product_name  # fallback

    scaled_value = scaler.transform([[encoded_value]])

    try:
        predicted_label = int(model.predict(scaled_value)[0])
    except:
        return product_name

    same_category = products_df[products_df["label"] == predicted_label]

    if same_category.empty:
        current_price = products_df[products_df["product"] == product_name]["price"].values[0]
        diff = (products_df["price"] - current_price).abs()
        return products_df.iloc[diff.idxmin()]["product"]

    filtered = same_category[same_category["product"] != product_name]
    if filtered.empty:
        filtered = same_category

    return filtered.sample(1).iloc[0]["product"]

# ----------------------------------------------------
# SIDEBAR – CART
# ----------------------------------------------------
with st.sidebar:
    st.header("🛍️ Your Cart")

    if len(st.session_state.cart) == 0:
        st.info("Cart is empty.")
    else:
        total = 0
        for item in st.session_state.cart:
            price = products_df[products_df["product"] == item]["price"].values[0]
            total += price
            st.write(f"✔ **{item}** – ₹{price}")

        st.success(f"### Total: ₹{total}")

    if st.button("Clear Cart"):
        st.session_state.cart = []
        st.rerun()

# ----------------------------------------------------
# SEARCH BAR
# ----------------------------------------------------
st.subheader("🔍 Search for products")
search = st.text_input("Type product name…")

# ----------------------------------------------------
# PRODUCT LIST
# ----------------------------------------------------
st.subheader("📦 Available Products")

filtered = (
    products_df[products_df["product"].str.contains(search, case=False)]
    if search else products_df
)

cols = st.columns(3)
i = 0

for _, row in filtered.iterrows():
    with cols[i % 3]:
        st.markdown(f"### {row['product']}")
        st.write(f"**Price:** ₹{row['price']}")

        if st.button(f"Add to Cart 🛒 ({row['product']})"):
            add_to_cart(row["product"])
            st.rerun()

    i += 1

# ----------------------------------------------------
# RECOMMENDATIONS SECTION
# ----------------------------------------------------
st.subheader("🤖 Smart ML Recommendations")

if len(st.session_state.cart) == 0:
    st.info("Add items to get AI-powered recommendations.")
else:
    last_item = st.session_state.cart[-1]
    st.write(f"Based on your last item **{last_item}**, you may also like:")

    try:
        recommended_item = get_recommendation(last_item)
        st.success(f"### ⭐ Recommended: {recommended_item}")
    except:
        st.warning("Recommendation model could not process this item.")
