from flask import Flask, request, jsonify
from app.service import get_prediction, get_stock_data, calculate_technical_indicators
from datetime import datetime
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route('/get-predict', methods=['GET'])
    def get_predict():
        stock_code = request.args.get('stock_code')
        number_of_days = request.args.get('days', default=30, type=int)

        if not stock_code:
            return jsonify({"error": "Mã cổ phiếu không được cung cấp"}), 400

        predict_data = get_prediction(stock_code=stock_code, number_of_days=number_of_days)

        if predict_data is None:
            return jsonify({"error": "Không thể dự đoán cho mã cổ phiếu này"}), 500

        return jsonify({"stock_code": stock_code, "predictions": predict_data})

    @app.route('/get-stock', methods=['GET'])
    def get_stock():
        stock_code = request.args.get('stock_code')
        start_date = request.args.get('start', default='2020-01-01')
        end_date = request.args.get('end', default=datetime.now().strftime('%Y-%m-%d'))
        interval_type = request.args.get('interval_type', default='1D')

        # Kiểm tra mã cổ phiếu
        if not stock_code:
            return jsonify({"error": "Mã cổ phiếu không được cung cấp"}), 400

        # Kiểm tra loại interval hợp lệ
        valid_intervals = ['1m', '5m', '15m', '30m', '1H', '1D', '1W', '1M']
        if interval_type not in valid_intervals:
            return jsonify({
                "error": "Loại khoảng thời gian không hợp lệ",
                "valid_types": valid_intervals
            }), 400

        # Gọi hàm để lấy dữ liệu cổ phiếu
        stock_data = get_stock_data(
            stock_code=stock_code, 
            start=start_date, 
            end=end_date,
            interval=interval_type
        )

        if stock_data is None:
            return jsonify({"error": "Không thể lấy dữ liệu cho mã cổ phiếu này"}), 500

        return jsonify({
            "stock_code": stock_code,
            "interval": interval_type,
            "data": stock_data
        })

    @app.route('/get-indicators', methods=['GET'])
    def get_indicators():
        stock_code = request.args.get('stock_code')
        if not stock_code:
            return jsonify({"error": "Mã cổ phiếu không được cung cấp"}), 400

        # Chỉ lấy 14 ngày gần nhất
        end_date = datetime.now().strftime('%Y-%m-%d')
        result = calculate_technical_indicators(stock_code=stock_code)

        if result is None:
            return jsonify({"error": "Không thể tính toán chỉ số cho mã cổ phiếu này"}), 500

        return jsonify({
            "stock_code": stock_code,
            "indicators": result
        })

    return app
