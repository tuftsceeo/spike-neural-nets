import Hub 
import asyncio

async def main():
    spike = Hub.Hub_PS(hub = 0)  # hub0 = spike
    print("asking")
    await spike.ask()


if __name__ == "__main__":
    asyncio.run(main())