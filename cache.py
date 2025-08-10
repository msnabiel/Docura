import os
import hashlib
import pickle
import time
from typing import Optional, Dict, Any

class DocumentCache:
    """Cache manager for processed documents to avoid reprocessing"""
    
    def __init__(self, cache_dir: str, logger):
        self.cache_dir = cache_dir
        self.logger = logger
        self.cache_stats = {"hits": 0, "misses": 0, "saves": 0}
        
        os.makedirs(cache_dir, exist_ok=True)
        self.logger.info(f"Document cache initialized at {cache_dir} (no expiry)")

    def _get_cache_key(self, source: str) -> str:
        if source.startswith(('http://', 'https://')):
            return hashlib.md5(source.encode()).hexdigest()
        try:
            stat = os.stat(source)
            key_data = f"{source}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except:
            return hashlib.md5(source.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _is_cache_valid(self, cache_path: str) -> bool:
        return os.path.exists(cache_path)

    def get(self, source: str) -> Optional[Dict[str, Any]]:
        cache_key = self._get_cache_key(source)
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                self.cache_stats["hits"] += 1
                self.logger.info(f"Cache HIT for {source} (key: {cache_key})")
                return cached_data
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {source}: {e}")
        
        self.cache_stats["misses"] += 1
        self.logger.info(f"Cache MISS for {source} (key: {cache_key})")
        return None

    def set(self, source: str, data: Dict[str, Any]) -> bool:
        cache_key = self._get_cache_key(source)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cached_data = {
                "data": data,
                "source": source,
                "cache_key": cache_key,
                "cached_at": time.time()
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            self.cache_stats["saves"] += 1
            self.logger.info(f"Cache SAVED for {source} (key: {cache_key})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save cache for {source}: {e}")
            return False

    def clear_expired(self) -> int:
        self.logger.info("No cache expiry configured - no entries to clear")
        return 0

    def get_stats(self) -> Dict[str, Any]:
        try:
            cache_size = len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')])
        except:
            cache_size = 0
        
        hit_rate = 0
        if self.cache_stats["hits"] + self.cache_stats["misses"] > 0:
            hit_rate = self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"])
        
        return {
            "cache_dir": self.cache_dir,
            "cache_size": cache_size,
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "saves": self.cache_stats["saves"],
            "hit_rate": f"{hit_rate:.2%}",
            "expiry": "None (permanent cache)"
        }

    def clear_all(self) -> bool:
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
            self.logger.info("All cache entries cleared")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing all cache: {e}")
            return False
