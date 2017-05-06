REMOTE=gpu0.qed.ai
REMOTE_DIR=dl1617-mnist
VENV=dl1617-mnist-venv

rsync -axv lab6.py $REMOTE:$REMOTE_DIR/

RUN2="source $VENV/bin/activate && cd $REMOTE_DIR && python lab6.py || bash"
RUN1="tmux new \"$RUN2\""
echo $RUN1
ssh -t $REMOTE "$RUN1"
