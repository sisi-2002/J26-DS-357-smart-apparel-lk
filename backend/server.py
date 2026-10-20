"""
Flask JSON API for the Smart Apparel LK Component 4 React frontend.

Expected frontend request:
POST /api/predict
Content-Type: application/json

{
  "order_quantity": 10000,
  "handover_days": 10,
  "required_smv_total": 12.5,
  "order_value": 2500000,
  "available_lines": 4,
  "style": "Polo",
  "priority": "Normal"
}

The API returns JSON (not a Jinja-rendered HTML page) so the React
ResultsPanel can consume the result object directly.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from factory_config import FACTORY_SHIFTS_PER_DAY
from historical_capacity import build_historical_capacity_plan
from line_model_features import (
    FEASIBILITY_MODEL_TARGET,
    LINE_MODEL_FEATURES,
    historical_line_count,
    make_line_features,
)
from research_analysis import (
    analyze_intervention_options,
    predict_line_feasibility,
)

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

ALLOWED_STYLES = {"Polo", "T-Shirt", "Hoodie", "Legging", "Jacket"}
ALLOWED_PRIORITIES = {"Low", "Normal", "High"}
ALLOWED_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}


def to_json_safe(value):
    """Convert NumPy/Pandas values and nested containers to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@app.after_request
def add_local_frontend_cors(response):
    """Permit the local Vite frontend to call this JSON API."""
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def load_bundle(path):
    """Load a model bundle; return None if the optional model is unavailable."""
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        if isinstance(bundle, dict) and bundle.get("type") == "keras":
            import tensorflow as tf
            bundle["model"] = tf.keras.models.load_model(
                bundle["model_path"], compile=False
            )
        return bundle
    except Exception as exc:
        app.logger.exception("Could not load model bundle %s: %s", path, exc)
        return None


# Keep the same historical datasets and model artifacts used by the existing app.py.
try:
    orders_data = pd.read_csv(BASE_DIR / "data" / "orders.csv")
    interventions_data = pd.read_csv(BASE_DIR / "data" / "interventions.csv")
except (OSError, pd.errors.ParserError) as exc:
    raise RuntimeError(
        "Could not load data/orders.csv or data/interventions.csv. "
        "Keep the data folder beside server.py."
    ) from exc

if "shift_hours" not in orders_data.columns:
    raise RuntimeError("data/orders.csv must contain a 'shift_hours' column.")

shift_hours = float(pd.to_numeric(
    orders_data["shift_hours"], errors="coerce"
).median())
if not np.isfinite(shift_hours) or shift_hours <= 0:
    raise RuntimeError("Could not determine a valid shift_hours value from orders.csv.")

capacity_model = load_bundle(BASE_DIR / "models" / "line_shift_capacity_model.pkl")
line_model = load_bundle(BASE_DIR / "models" / "line_feasibility_model.pkl")

if line_model and line_model.get("target") != FEASIBILITY_MODEL_TARGET:
    # Avoid treating an older on-time model as an Accept/Reject classifier.
    line_model = None

if line_model:
    required_columns = {
        "n_shifts", "order_quantity", "handover_days", "required_smv_total",
        "order_value", "style_id", "priority", "n_available_workers", "shift_hours",
    }
    if required_columns.issubset(orders_data.columns):
        training_orders = orders_data[
            pd.to_numeric(orders_data["n_shifts"], errors="coerce")
            .le(FACTORY_SHIFTS_PER_DAY)
        ]
        if not training_orders.empty:
            line_features = pd.DataFrame(
                [
                    make_line_features(
                        row.order_quantity,
                        row.handover_days,
                        row.required_smv_total,
                        row.order_value,
                        row.style_id,
                        row.priority,
                        historical_line_count(row.n_available_workers),
                        row.n_shifts,
                        row.shift_hours,
                    )
                    for row in training_orders.itertuples()
                ],
                columns=LINE_MODEL_FEATURES,
            )
            for feature in LINE_MODEL_FEATURES:
                limits = [
                    float(line_features[feature].min()),
                    float(line_features[feature].max()),
                ]
                line_model.setdefault("training_ranges", {}).setdefault(
                    feature, limits
                )

n2_bundle = load_bundle(BASE_DIR / "models" / "best_n2_model.pkl")
cost_model = load_bundle(BASE_DIR / "models" / "best_cost_model.pkl")

training_report = {}
try:
    with (BASE_DIR / "models" / "model_comparison.json").open(
        encoding="utf-8"
    ) as report_file:
        training_report = json.load(report_file)
except (OSError, json.JSONDecodeError):
    app.logger.warning("models/model_comparison.json is unavailable or invalid.")


@app.get("/")
def health():
    """Simple health check for the backend."""
    return jsonify({
        "status": "ok",
        "service": "Smart Apparel LK Component 4 API",
        "prediction_endpoint": "/api/predict",
    })


@app.route("/api/predict", methods=["POST", "OPTIONS"])
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Validate the React request and return the complete assessment as JSON."""
    if request.method == "OPTIONS":
        return ("", 204)

    if not request.is_json:
        return jsonify({
            "error": "Request body must be JSON with Content-Type: application/json."
        }), 415

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        qty = int(payload.get("order_quantity"))
        days = int(payload.get("handover_days"))
        available_lines = int(payload.get("available_lines"))
        smv = float(payload.get("required_smv_total"))
        order_value = float(payload.get("order_value"))
    except (TypeError, ValueError, OverflowError):
        return jsonify({
            "error": (
                "Enter valid numeric values for order_quantity, handover_days, "
                "available_lines, required_smv_total and order_value."
            )
        }), 400

    style = payload.get("style")
    priority = payload.get("priority")

    if qty <= 0 or days <= 0 or smv <= 0 or order_value <= 0:
        return jsonify({
            "error": (
                "order_quantity, handover_days, required_smv_total and "
                "order_value must be greater than zero."
            )
        }), 400
    if not 1 <= available_lines <= 30:
        return jsonify({
            "error": "available_lines must be an integer from 1 to 30."
        }), 400
    if style not in ALLOWED_STYLES:
        return jsonify({
            "error": "style must be one of: Polo, T-Shirt, Hoodie, Legging, Jacket."
        }), 400
    if priority not in ALLOWED_PRIORITIES:
        return jsonify({
            "error": "priority must be one of: Low, Normal, High."
        }), 400

    # The existing factory configuration fixes the factory to its configured shifts.
    available_shifts = FACTORY_SHIFTS_PER_DAY

    try:
        production_plan = build_historical_capacity_plan(
            orders_data,
            qty,
            days,
            smv,
            style,
            available_lines,
            available_shifts,
            shift_hours,
            order_value=order_value,
            priority=priority,
            capacity_model=capacity_model,
        )
        if production_plan.get("error"):
            return jsonify({"error": str(production_plan["error"])}), 422

        n1_analysis = predict_line_feasibility(
            line_model,
            training_report.get("line_model_metrics"),
            qty,
            days,
            smv,
            order_value,
            style,
            priority,
            available_lines,
            available_shifts,
            shift_hours,
        )

        cost_metrics = training_report.get("cost_models", {}).get(
            training_report.get("cost_best"), {}
        )
        n2_analysis = analyze_intervention_options(
            n2_bundle,
            cost_model,
            training_report.get("N2_models"),
            cost_metrics,
            interventions_data,
            orders_data,
            qty,
            days,
            smv,
            order_value,
            style,
            priority,
        )

        recommended = production_plan["recommended"]
        decision = production_plan["decision"]
        explanation = [
            (
                f"{production_plan['capacity_source']}: estimated "
                f"{production_plan['rate_median']:.1f} pieces per line-shift "
                "per day at this order's SMV."
            ),
            (
                f"The median plan needs "
                f"{production_plan['plans'][0]['line_shifts_required']} line-shifts; "
                f"the conservative plan needs "
                f"{production_plan['plans'][1]['line_shifts_required']}. "
                f"Maximum capacity is {production_plan['max_line_shifts']} "
                f"line-shifts across {available_lines} lines and "
                f"{available_shifts} shifts."
            ),
            (
                f"Estimated completion: {recommended['estimated_days']} days "
                f"at the expected rate; conservative estimate: "
                f"{recommended['conservative_days']} days."
            ),
            str(production_plan.get("planning_note", "")),
            str(production_plan.get("machine_note", "")),
        ]

        # No live named-worker allocation endpoint exists in the current backend.
        # Be explicit so the UI does not imply these are real-time assignments.
        workforce = {
            "available": False,
            "message": (
                "The current backend has no live worker allocation endpoint. "
                "The displayed staffing plan is a required grade mix, not "
                "named worker recommendations."
            ),
        }

        # Keep these top-level keys stable for the React ResultsPanel.
        response = {
            "order_quantity": qty,
            "handover_days": days,
            "estimated_smv": smv,
            "estimated_order_value": order_value,
            "estimated_workload": qty * smv,
            "capacity_ratio": float(recommended["capacity_ratio"]),
            "style": style,
            "priority": priority,
            "production_plan": production_plan,
            "n1_analysis": n1_analysis,
            "n2_analysis": n2_analysis,
            "decision": decision,
            "explanation": explanation,
            "workforce": workforce,
        }
        return jsonify(to_json_safe(response)), 200

    except Exception:
        app.logger.exception("Order prediction failed.")
        return jsonify({
            "error": (
                "The order assessment could not be completed. Check the backend "
                "terminal for details and verify that the required model files "
                "and data are available."
            )
        }), 500


if __name__ == "__main__":
    # Development server. For production, use a WSGI server instead.
    app.run(host="0.0.0.0", port=5000, debug=True)
