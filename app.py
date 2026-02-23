from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")
CORS(app)

# ── Load model + scaler ──────────────────────────────────────────
model  = pickle.load(open("mumbai_price_model.pkl", "rb"))
scaler = pickle.load(open("mumbai_scaler.pkl",      "rb"))

# ── Load dataset for locality list + median price_per_sqft ──────
df = pd.read_csv("mumbai_house_price.csv")
df['locality'] = df['locality'].astype(str).str.strip().str.title()

locality_corrections = {
    'Dombivli':        'Dombivali',
    'Dombivli W':      'Dombivali',
    'Vile Parle East': 'Vile Parle',
    'Nallasopara W':   'Nalasopara',
    'Nalasopara':      'Nalasopara'
}
df['locality'] = df['locality'].replace(locality_corrections)

# Clean outliers (same as training)
df = df[(df['price']          >  1_000_000) & (df['price']          < 100_000_000)]
df = df[(df['price_per_sqft'] >      2_000) & (df['price_per_sqft'] <      40_000)]
df = df[(df['area']           >        250) & (df['area']           <       5_000)]

localities         = sorted(df['locality'].dropna().unique().tolist())
locality_price_map = df.groupby('locality')['price_per_sqft'].median().to_dict()
global_median_ppsf = float(df['price_per_sqft'].median())


# ── Helpers ──────────────────────────────────────────────────────
def format_indian_price(price: float) -> str:
    price = int(price)
    crore = price // 10_000_000
    lakh  = (price % 10_000_000) // 100_000
    if crore > 0:
        if lakh > 0:
            return f"₹{crore} Crore {lakh} Lakh"
        return f"₹{crore} Crore"
    elif lakh > 0:
        return f"₹{lakh} Lakh"
    else:
        return f"₹{price:,}"


# ── Page routes ──────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict-page')
def predict_page():
    return render_template("predict.html")

@app.route('/insights')
def insights_page():
    return render_template("insights.html")

@app.route('/model-page')
def model_page():
    return render_template("model.html")


# ── API routes ───────────────────────────────────────────────────
@app.route('/localities')
def get_localities():
    return jsonify(localities)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        area         = float(data['area'])
        bedroom_num  = int(data['bedroom_num'])
        bathroom_num = int(data['bathroom_num'])
        balcony_num  = int(data['balcony_num'])
        age          = float(data['age'])
        total_floors = int(data['total_floors'])
        locality     = str(data['locality']).strip().title()
        locality     = locality_corrections.get(locality, locality)

        price_per_sqft = locality_price_map.get(locality, global_median_ppsf)

        features        = np.array([[area, bedroom_num, bathroom_num,
                                      balcony_num, age, total_floors, price_per_sqft]])
        features_scaled = scaler.transform(features)
        predicted_price = float(model.predict(features_scaled)[0])
        predicted_price = max(predicted_price, 0)

        return jsonify({
            "predicted_price": round(predicted_price, 2),
            "formatted_price": format_indian_price(predicted_price),
            "locality":        locality,
            "price_per_sqft":  round(price_per_sqft, 0),
            "model":           "Random Forest (R²: 0.9995)"
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
