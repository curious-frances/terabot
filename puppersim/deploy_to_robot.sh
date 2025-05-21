#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  REMOTE=$(avahi-resolve-host-name raspberrypi.local -4 | awk '{print $2}')
elif [[ "$OSTYPE" == "darwin"* ]]; then
  REMOTE=192.168.2.3 #raspberrypi.local
else
  print "not supported os"
  exit 1
fi

echo PI addrees is "${REMOTE}"
REMOTE_DIR=puppersim_deploy

ssh frances@${REMOTE} mkdir -p ${REMOTE_DIR}
rsync -avz ${PWD} frances@${REMOTE}:${REMOTE_DIR}

if [ -z "$1" ]; then
  ssh -t frances@${REMOTE} "cd ${REMOTE_DIR} ; bash --login"
else
  ssh -t frances@${REMOTE} "cd ${REMOTE_DIR} ; $@"
fi
