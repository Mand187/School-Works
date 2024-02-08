package cache

import "errors"

type Cacher[K comparable, V any] interface {
	Get(key K) (value V, err error)
	Put(key K, value V) (err error)
}

// Concrete LRU cache
type lruCache[K comparable, V any] struct {
	size      int
	remaining int
	cache     map[K]V
	queue     []K
}

// Constructor
func NewCacher[K comparable, V any](size int) Cacher[K, V] {
	return &lruCache[K, V]{size: size, remaining: size, cache: make(map[K]V), queue: make([]K, 0)}
}

// Helper method to delete all occurrences of a key from the queue
func (c *lruCache[K, V]) deleteFromQueue(key K) {
	newQueue := make([]K, 0, c.size)
	for _, k := range c.queue {
		if k != key {
			newQueue = append(newQueue, k)
		}
	}
	c.queue = newQueue
}

// Helper method to move a key to the tail of the queue
func (c *lruCache[K, V]) moveToTail(key K) {
	c.deleteFromQueue(key)
	c.queue = append(c.queue, key)
}

func (c *lruCache[K, V]) Get(key K) (value V, err error) {
	// Check if the key exists in the cache
	if val, ok := c.cache[key]; ok {
		// Move the key to the tail of the queue (mark as recently used)
		c.moveToTail(key)
		return val, nil
	}
	// Key not found, return the zero value of V
	return value, errors.New("key not found")
}

func (c *lruCache[K, V]) Put(key K, value V) (err error) {
	// Check if the key already exists in the cache
	if _, ok := c.cache[key]; ok {
		c.cache[key] = value
		c.moveToTail(key)
		return nil
	}

	// Check capacity and evict if needed
	if c.remaining == 0 {
		removedKey := c.queue[0]
		delete(c.cache, removedKey)
		c.deleteFromQueue(removedKey)
		c.remaining++
	}

	// Add the new key-value pair
	c.cache[key] = value
	c.queue = append(c.queue, key)
	c.remaining--

	return nil
}
