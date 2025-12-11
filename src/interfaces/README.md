# Interfaces Layer

The interfaces package hosts glue code that wires services and infrastructure together for consumption by apps. Examples include:

- Dependency containers or providers that construct services from configuration.
- Adapters translating between FastAPI dependencies and service instances.
- Message bus or event dispatcher registrations.

Interfaces should remain thin and declarative, keeping business logic inside the `services` layer.
