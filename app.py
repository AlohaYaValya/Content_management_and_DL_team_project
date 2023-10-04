from flask import Flask, render_template, session, g, request
from forms import *
from medical_llama import *


app = Flask(__name__)


# import config
# app.config.from_object(config)
# db.init_app(app)
# 迁移数据库
# migrate = Migrate(app, db)
# orm 映射成表
# 1. 初始化  flask db init
# 2. 每次更新生成脚本 flask db migrate
# 3. 同步数据库 flask db upgrade
# before_request
# 模块化
# app.register_blueprint(qa_bp)
# app.register_blueprint(auth_bp)
# app.register_blueprint(nmt_bp)

@app.route('/')
def index():  # put application's code here
    return render_template("base.html")


@app.route("/question", methods=['GET', 'POST'])
def question():
    if request.method == 'GET':
        return render_template("question.html", answer="")
    else:
        form = DialForm(request.form)
        if form.content.data == "":
            return render_template("question.html", answer="")
        else:
            answer = form.content.data + '\n'
            sentence = form.content.data

            answer = response(sentence).split("Response:")[1].strip()

            # print(answer)
            return render_template("question.html", answer=answer)
    #


if __name__ == '__main__':
    app.run(port=7860)
