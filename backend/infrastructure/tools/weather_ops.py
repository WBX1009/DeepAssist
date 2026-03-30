def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气情况。
    :param city: 城市名称，例如 "北京"、"Tokyo"
    """
    # 这里我们用占位实现。实际接入时可调用 Open-Meteo 等免费 API。
    # 我们关注的是 Agent 能否准确提取参数并调用该函数。
    mock_data = {
        "北京": "晴朗，气温 22°C，微风。",
        "上海": "多云转小雨，气温 18°C。",
        "Tokyo": "樱花季，晴朗，气温 20°C。"
    }
    weather = mock_data.get(city, f"暂无 {city} 的天气数据，请尝试其他大城市。")
    return weather