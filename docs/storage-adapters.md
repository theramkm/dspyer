# Storage Adapters

`dspyer` features thread-safe pluggable logging storage adapters to record runtime metrics, compile validation reports, and capture datasets for offline training.

---

## 1. Pluggable Storage Adapters

To prevent performance bottlenecks under concurrent LLM calls (e.g. FastAPI request handling), logging writes are delegated to a storage adapter interface. 

By default, `dspyer` uses [FileStorageAdapter](https://github.com/theramkm/dspyer/blob/main/dspyer/utils.py) which appends lines to local files asynchronously in thread pools using `asyncio.to_thread`.

---

## 2. Creating a Custom Storage Adapter

You can redirect logs to external databases (e.g. SQLite, PostgreSQL, MongoDB, or vector databases) by subclassing the [BaseStorageAdapter](https://github.com/theramkm/dspyer/blob/main/dspyer/utils.py) and registering it:

```python
from dspyer.utils import BaseStorageAdapter, set_storage_adapter

class MongoDBStorageAdapter(BaseStorageAdapter):
    def append_line(self, target: str, line: str) -> None:
        # target represents the log filepath/identifier
        db[target].insert_one({"payload": line})

    async def append_line_async(self, target: str, line: str) -> None:
        await db_client.async_insert(target, line)

# Register the adapter globally
set_storage_adapter(MongoDBStorageAdapter())
```
