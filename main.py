from flask import Flask
from flask import request
from parlai.core.agents import create_agent
from parlai.core.opt import Opt

app = Flask(__name__)

opt = Opt(
    model_file = 'model',
    interactive_task = True,
    task = 'interactive',
    interactive_mode=True,
    override = {
        'fp16' : True,
        'beam_context_block_ngram' : 3,
        'beam_block_ngram' : 3,
        'inference' : 'topk',
        'topk' : 40,
        'beam_size' : 10,
        'beam_min_length' : 10,
        'temperature' : 1.0
    }
)

agent = create_agent(opt, requireModelExists=True)
agent.opt.log()


@app.route("/", methods=["GET", "POST"])
def send_msg():

    content = request.get_json()

    utterance = content["data"]

    reply = {'episode_done': False, 'text': utterance}

    agent.observe(reply)

    model_res = agent.act()

    return model_res["text"]


if __name__ == '__main__':
    app.run(host='0.0.0.0')