import requests
from backend.common.logger import get_logger

logger = get_logger(__name__)

def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气情况。
    :param city: 城市名称，例如 "北京"、"Tokyo"、"New York"
    """
    logger.info(f"🛠️ [Tool] 正在调用真实天气 API 查询: {city}")
    try:
        # 第一步：地理编码 (将城市名转换为经纬度)
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1, "language": "zh"}
        geo_resp = requests.get(geo_url, params=geo_params, timeout=5).json()
        
        if "results" not in geo_resp or not geo_resp["results"]:
            return f"查询失败：无法在地图上定位城市 '{city}'。请尝试使用更大的城市名称或检查拼写。"
            
        location = geo_resp["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        resolved_city = location.get("name", city)
        
        # 第二步：获取实时天气
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "timezone": "auto"
        }
        weather_resp = requests.get(weather_url, params=weather_params, timeout=5).json()
        
        if "current_weather" not in weather_resp:
            return "查询失败：天气服务暂时不可用，请稍后再试。"
            
        current = weather_resp["current_weather"]
        temp = current.get("temperature", "未知")
        wind = current.get("windspeed", "未知")
        
        # 简单的天气代码映射 (Open-Meteo WMO Code)
        code = current.get("weathercode", 0)
        weather_desc = "晴朗/多云" if code <= 3 else "雾/降水" if code <= 69 else "雪/雷暴"
        
        return f"📍 城市: {resolved_city}\n🌡️ 当前气温: {temp}°C\n💨 风速: {wind} km/h\n🌤️ 概况: {weather_desc}"
        
    except requests.RequestException as e:
        logger.error(f"天气 API 网络异常: {e}")
        return f"网络异常，无法获取天气数据: {str(e)}"
    except Exception as e:
        logger.error(f"天气查询工具内部错误: {e}")
        return f"工具内部错误: {str(e)}"