from flask import Flask, request, render_template
import logging
import logging.config
import yaml
from kh import predict

app = Flask(__name__, static_url_path='')

try:
    with open('logging.yml') as fd:
        conf = yaml.load(fd)
        logging.config.dictConfig(conf['logging'])
except OSError:
    conf = None

logger = logging.getLogger('app')
input_logger = logging.getLogger('app.input')

if conf:
    logger.info('logging.yml found, applying config')
    logger.debug(conf)
else:
    logger.info('logging.yml not found')


from uuid import uuid4

@app.route('/')
def root():
    uuid = request.cookies.get('uuid', uuid4())
    resp = app.send_static_file('index.html')
    resp.set_cookie('uuid', str(uuid))
    return resp

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    rec = {'ip': ip(),
           'uuid': request.cookies.get('uuid'),
           'data': request.form.get('in')}
    input_logger.info(rec)
    return predict(request.form.get('in'))

def ip():
    return request.environ.get('REMOTE_ADDR', request.remote_addr)

if __name__ == '__main__':
    app.run()

