ps -ef | grep tensorboard | grep -v grep | awk '{print $2}' | xargs kill
