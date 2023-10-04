import wtforms


class DialForm(wtforms.Form):
    content = wtforms.StringField()
