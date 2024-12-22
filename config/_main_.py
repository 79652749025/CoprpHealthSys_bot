if __name__ == '__main__':
    import nest_asyncio
    nest_asyncio.apply()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
