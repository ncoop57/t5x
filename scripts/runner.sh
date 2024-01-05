POD_NAME="stable-t5x-test"

bash send.sh $POD_NAME setup.sh
bash run.sh $POD_NAME "bash setup.sh"
bash run.sh $POD_NAME "source env-t5x/bin/activate; cd stable_t5; bash stable_t5.sh"