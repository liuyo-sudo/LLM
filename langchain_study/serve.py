import asyncio
import json
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import redis.asyncio as redis
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import aioredlock
from dotenv import load_dotenv
import os
import uvicorn
import mysql.connector
from urllib.parse import unquote
import time
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # 加载环境变量
load_dotenv()

# 配置
MYSQL_URL = "mysql+mysqlconnector://root:BW%40lyb3210@localhost:3306/langchain_db"
REDIS_URL = "rd://localhost:6379"
SECRET_KEY = "123456"


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# FastAPI 应用
app = FastAPI(title="LangChain 多智能体系统")

# Redis 客户端，带连接池
redis_pool = redis.ConnectionPool.from_url(REDIS_URL, max_connections=100)
redis_client = redis.Redis.from_pool(redis_pool)
redis_lock = aioredlock.Aioredlock([redis_client])

lock_key = "my_lock"
lock_timeout = 10  # 锁的超时时间，单位秒


def acquire_lock(lock_key, timeout=10):
    """尝试获取锁"""
    end_time = time.time() + timeout
    while time.time() < end_time:
        if redis_client.setnx(lock_key, 1):
            redis_client.expire(lock_key, timeout)  # 设置锁的过期时间
            return True
        time.sleep(0.01)  # 短暂休眠后重试
    return False
#
# def release_lock(lock_key):
#     """释放锁"""
#     pipe = redis_client.pipeline(transaction=True)
#     while True:
#         try:
#             # 使用WATCH来确保在执行DEL命令前锁没有被其他客户端改变
#             pipe.watch(lock_key)
#             if pipe.get(lock_key):  # 检查锁是否存在且值未被篡改
#                 pipe.multi()
#                 pipe.delete(lock_key)
#                 pipe.execute()
#                 break
#             else:
#                 # 如果锁不存在或值被篡改，则退出循环
#                 break
#         except Exception as e:
#             # 如果WATCH监视的键在执行MULTI命令前被改变，则重试此操作
#             continue
#         finally:
#             pipe.reset()  # 重置管道以供下次使用
# MySQL 设置，带连接池
def create_database_if_not_exists():
    try:
        # 从 MYSQL_URL 提取连接信息
        user = MYSQL_URL.split("://")[1].split(":")[0]
        password = unquote(MYSQL_URL.split("://")[1].split(":")[1].split("@")[0])  # 解码密码
        host = MYSQL_URL.split("@")[1].split(":")[0]
        port = int(MYSQL_URL.split("@")[1].split(":")[1].split("/")[0])  # 转换为整数
        dbname = MYSQL_URL.split("/")[-1]

        # 不指定数据库连接
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password='BW@lyb3210'
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {dbname}")
        cursor.close()
        conn.close()
        logger.info(f"数据库 {dbname} 已创建或已存在")
    except mysql.connector.Error as e:
        logger.error(f"连接 MySQL 或创建数据库失败: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"创建数据库时发生未知错误: {str(e)}")
        raise


# 在初始化引擎前创建数据库
create_database_if_not_exists()

engine = create_engine(
    MYSQL_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30
)
Base = declarative_base()


# SQLAlchemy 模型
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String(255))


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String(36), index=True)
    message = Column(Text)
    role = Column(String(20))
    timestamp = Column(DateTime, default=datetime.utcnow)


class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String(100))
    config = Column(Text)  # JSON 格式的 Agent 配置
    created_at = Column(DateTime, default=datetime.utcnow)

class Workflow(Base):
    __tablename__ = "workflows"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    agent_id = Column(Integer, ForeignKey("agents.id"))
    name = Column(String(100))
    config = Column(Text)  # JSON 格式的 Workflow 配置
    created_at = Column(DateTime, default=datetime.utcnow)

try:
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表创建成功")
except OperationalError as e:
    logger.error(f"创建表失败: {str(e)}")
    raise
except Exception as e:
    logger.error(f"创建表时发生未知错误: {str(e)}")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Pydantic 模型
class UserCreate(BaseModel):
    username: str
    password: str


class AgentConfig(BaseModel):
    name: str
    plugins: List[str]
    workflow: Dict[str, Any]
    rag_config: Dict[str, Any]
    code_snippets: List[str]


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str
    api_key: str
    base_url: str

class WorkFlowConfig(BaseModel):
    name: str
    plugins: List[str]
    llm: Dict[str, Any]
    rag_config: Dict[str, Any]
    code_snippets: List[str]
# 安全设置
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# 依赖注入
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="无效的认证凭据")
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise HTTPException(status_code=401, detail="无效的认证凭据")
        return user
    except Exception as e:
        logger.error(f"认证错误: {str(e)}")
        raise HTTPException(status_code=401, detail="无效的认证凭据")


# 辅助函数
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict):
    to_encode = data.copy()
    now = datetime.now()

    ts = datetime.timestamp(now)
    td = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES).seconds
    ts = ts + td
    to_encode.update({"exp": ts})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# # FastAPI 生命周期事件
# @app.on_event("startup")
# async def startup_event():
#     try:
#         async with redis_client as conn:
#             await conn.ping()
#         logger.info("成功连接到 Redis")
#     except rd.RedisError as e:
#         logger.error(f"连接 Redis 失败: {str(e)}")
#         raise
#     except Exception as e:
#         logger.error(f"连接 Redis 时发生未知错误: {str(e)}")
#         raise
#     try:
#         with engine.connect() as conn:
#             conn.execute("SELECT 1")
#         logger.info("成功连接到 MySQL")
#     except Exception as e:
#         logger.error(f"连接 MySQL 失败: {str(e)}")
#         raise
#
#
# @app.on_event("shutdown")
# async def shutdown_event():
#     await redis_client.close()
#     engine.dispose()
#     logger.info("连接已关闭")


# API 端点
@app.post("/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    await redis_client.setnx(lock_key, 1)
    try:
        #async with (await redis_lock.lock(f"user_register_{user.username}")):
            db_user = db.query(User).filter(User.username == user.username).first()
            if db_user:
                return {"msg": "用户名已被注册"}
                #raise HTTPException(status_code=400, detail="用户名已被注册")
            hashed_password = get_password_hash(user.password)
            db_user = User(username=user.username, hashed_password=hashed_password)
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            logger.info(f"用户 {user.username} 注册成功")
            return {"msg": "用户创建成功"}
    # except aioredlock.LockError as e:
    #     logger.error(f"获取 Redis 锁 user_register_{user.username} 失败: {str(e)}")
    #     release_lock(lock_key)
    #     raise HTTPException(status_code=500, detail="获取锁失败")
    except Exception as e:
        logger.error(f"注册错误: {str(e)}")
        await redis_client.delete(lock_key)
        raise HTTPException(status_code=500, detail="服务器内部错误")
    finally:
        await redis_client.delete(lock_key)
    return {"msg": "用户创建异常"}


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.username == form_data.username).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        access_token = create_access_token(data={"sub": user.username})
        logger.info(f"用户 {form_data.username} 登录成功")
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"登录错误: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@app.post("/chat")
async def chat(request: ChatRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    await redis_client.setnx(lock_key, 1)
    try:
        #async with (await redis_lock.lock(f"chat_{request.session_id}")):
            # 从 Redis 获取历史记录
        redis_key = f"chat:{user.id}:{request.session_id}"
        history_json = await redis_client.get(redis_key)
        history = []
        if history_json:
            his = json.loads(history_json)

            for item in his:
                history.append((item[0], item[1]))
        #model="deepseek-chat", api_key="sk-6e78fd3004da4265945815164509ec19",base_url="https://api.deepseek.com/"
        # 创建 LangChain 提示
        llm = ChatOpenAI(model=request.model, api_key=request.api_key,base_url=request.base_url)
        history.append(("human", request.message))
        prompt = ChatPromptTemplate.from_messages(history)

        # 获取响应
        response = await llm.ainvoke([HumanMessage(content=request.message)])

        # 更新历史记录
        history.append(("human", request.message))
        history.append(("assistant", response.content))

        # 保存到 Redis
        await redis_client.set(redis_key, json.dumps(history), ex=3600)  # 1小时过期

        ts = datetime.now()
        print(type(ts))
        # 保存到 MySQL
        db_history = ChatHistory(
            user_id=user.id,
            session_id=request.session_id,
            message=request.message,
            role="human",
            timestamp=ts
        )
        db.add(db_history)
        db_history = ChatHistory(
            user_id=user.id,
            session_id=request.session_id,
            message=response.content,
            role="assistant",
            timestamp=ts
        )
        db.add(db_history)
        db.commit()

        logger.info(f"用户 {user.id} 的会话 {request.session_id} 已处理")
        return {"response": response.content, "session_id": request.session_id}
    # except aioredlock.LockError as e:
    #     logger.error(f"获取 Redis 锁 chat_{request.session_id} 失败: {str(e)}")
    #     raise HTTPException(status_code=500, detail="获取锁失败")
    except Exception as e:
        logger.error(f"聊天错误: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")
    finally:
        await redis_client.delete(lock_key)
    return {"msg": "用户创建异常"}


@app.post("/agent")
async def create_agent(config: AgentConfig, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    await redis_client.setnx(lock_key, 1)
    try:
        #async with (await redis_lock.lock(f"agent_create_{user.id}_{config.name}")):
            db_agent = Agent(
                user_id=user.id,
                name=config.name,
                config=json.dumps(config.dict())
            )
            db.add(db_agent)
            db.commit()
            db.refresh(db_agent)
            logger.info(f"用户 {user.id} 创建 Agent {config.name}")
            return {"msg": f"Agent {config.name} 创建成功", "agent_id": db_agent.id}
    # except aioredlock.LockError as e:
    #     logger.error(f"获取 Redis 锁 agent_create_{user.id}_{config.name} 失败: {str(e)}")
    #     raise HTTPException(status_code=500, detail="获取锁失败")
    except Exception as e:
        logger.error(f"Agent创建错误: {str(e)}")
        await redis_client.delete(lock_key)
        raise HTTPException(status_code=500, detail="Agent创建错误")
    finally:
        await redis_client.delete(lock_key)
    return {"msg": "agent创建异常"}
@app.get("/agent/agentlist")
async def get_agent_list(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        agent_list = db.query(Agent).limit(10).all()
        res = json.loads(agent_list)
        return res
    except Exception as e:
        logger.error(f"获取Agent列表失败: {str(e)}")
        raise HTTPException(status_code = 500, detail= "获取Agent列表失败")

@app.post("/agent/{agent_id}/create_workflow")
async def create_agent_workflow(workflow : WorkFlowConfig, agent_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    await redis_client.setnx(lock_key, 1)
    try:
        db_agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == user.id).first()
        if not db_agent:
            raise HTTPException(status_code=404, detail="Agent 未找到")
        db_wf = Workflow(
            user_id = user.id,
            agent_id = agent_id,
            name = workflow.name,
            config = json.dumps(workflow.dict())
        )
        db.add(db_wf)
        db.commit()
        db.refresh(db_wf)
        logger.info(f"用户 {user.id} 创建 工作流 {workflow.name}")
        return {"msg": f"Agent {workflow.name} 创建成功", "agent_id": db_agent.id}
        return {"msg": "工作流创建成功"}
    except Exception as e:
        logger.error(f"创建工作流失败:{str(e)}")
        await redis_client.delete(lock_key)
        raise HTTPException(status_code = 500, detail="创建工作流失败")
    finally:
        await redis_client.delete(lock_key)
    return {"msg": "创建工作流失败"}

@app.get("/agent/{agent_id}/run")
async def run_agent(agent_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        db_agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == user.id).first()
        if not db_agent:
            raise HTTPException(status_code=404, detail="Agent 未找到")

        config = json.loads(db_agent.config)

        # 简单 Agent 执行（可根据插件/工作流扩展）
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        prompt = ChatPromptTemplate.from_template(
            f"执行 Agent {config['name']}，插件: {', '.join(config['plugins'])}"
        )
        response = await llm.ainvoke([HumanMessage(content="运行 Agent")])

        logger.info(f"用户 {user.id} 执行 Agent {agent_id}")
        return {"response": response.content, "agent_config": config}
    except Exception as e:
        logger.error(f"Agent 执行错误: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


# 服务主入口
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)