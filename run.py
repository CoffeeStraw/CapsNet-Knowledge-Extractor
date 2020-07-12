"""
Flask debug server runner
Author: Antonio Strippoli
"""
from flaskr import create_app

app = create_app()
app.run(debug=True)
