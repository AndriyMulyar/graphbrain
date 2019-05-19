# Graphbrain
self improving graph brain with conjecturing


### Docker Running Sage Container (for dev reference)

Run command in Sage container with appropriate mounting for re
```bash
docker run -v $(pwd):/home/sage/graphbrain/ -it -e PYTHONPATH='/home/sage/graphbrain/' sagemath/sagemath sage /home/sage/graphbrain/graph_conjecturing/graph_bootstrap_percolation/conjecture_bootstrap_good.py
```

### Database back-up

Single database
```bash
docker exec -u postgres graphbrain_db_1 pg_dump -Fc graphbrain > db.dump

docker exec -i -u postgres graphbrain_db_1 pg_restore -C -d postgres < db.dump

```


All databases

```bash
docker exec -u postgres graphbrain_db_1 pg_dumpall -c > graph_brain_back_up_all_$(date +%Y-%m-%d).dump

cat db.dump | docker exec -i graphbrain_db_1 psql -U postgres

```
