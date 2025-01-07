from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/detect')
def detection():
  return {'status': 'Closed'}

if __name__ == '__main__':
  app.run(debug=True)