# Graphbrain
self improving graph brain for conjecturing


### Database back-up

Single database
```bash
docker exec -u postgres graphbrain_db_1 pg_dump -Fc graphbrain > db.dump

docker exec -i -u postgres graphbrain_db_1 pg_restore -C -d postgres < db.dump

```


All databases

```bash
docker exec -u postgres graphbrain_db_1 pg_dumpall -c > back_up_all_$(date +%Y-%m-%d).dump

cat db.dump | docker exec -i graphbrain_db_1 psql -U postgres

```
