FROM heroku/miniconda

# Grab requirements.txt.
ADD ./webapp/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt

# Add our code
ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

RUN conda install opencv
RUN conda install Pillow
RUN conda install scikit-learn
RUN while read requirement; do conda install --yes $requirement; done < requirements.txt

CMD gunicorn app:app
