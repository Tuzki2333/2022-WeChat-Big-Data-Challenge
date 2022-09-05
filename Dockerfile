FROM tione-wxdsj.tencentcloudcr.com/base/pytorch:py38-torch1.9.0-cu111-1.0.0

WORKDIR /opt/ml/wxcode

COPY ./opensource_models ./opensource_models
COPY ./save ./save
COPY ./third_party ./third_party

COPY ./requirements.txt ./

RUN pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

COPY ./*.py ./
COPY ./start.sh ./

CMD sh -c "sh start.sh"
