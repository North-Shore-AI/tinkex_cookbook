import Config

# Test-specific config
config :logger, level: :warning

# Skip snakebridge generation in tests to avoid Python provisioning.
System.put_env("SNAKEBRIDGE_SKIP", "1")
config :snakebridge, auto_install: :never
