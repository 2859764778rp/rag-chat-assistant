from config import STORE_NAME, DOC_DIR, logger
from ingest import ingest_directory


def main():
    n = ingest_directory(DOC_DIR, store_name=STORE_NAME)
    logger.info("init_kb 完成：共 %d 个 chunk（store=%s）", n, STORE_NAME)
    print(f"知识库初始化完成：共 {n} 个 chunk（store={STORE_NAME}）")


if __name__ == "__main__":
    main()
