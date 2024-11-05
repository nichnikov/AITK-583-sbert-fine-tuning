"""https://elasticsearch-py.readthedocs.io/en/latest/async.html"""

import asyncio
import pymssql
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from src.config import logger
from src.data_types import TextsDeleteSample, ROW, ElasticSettings


class DataFromDB:
    """
    class for getting data from MS Server with specific SQL Query.
    """

    def __init__(self, **kwargs):
        conn = pymssql.connect(host=kwargs["server_host"],
                               port=1433,
                               user=kwargs["user_name"],
                               password=kwargs["password"],
                               charset='cp1251',
                               database="master")
        self.cursor = conn.cursor(as_dict=True)

    def fetch_from_db(self, sys_id: int, date: str):
        """
        return data (rows) from MS SQL DB Statistic on date (usually today)
        """
        today_str = "'" + str(date) + "'"
        self.cursor.execute("SELECT * FROM StatisticsRAW.[search].FastAnswer_RBD  "
                            "WHERE SysID = {} AND (ParentBegDate <= {} AND ParentEndDate IS NULL "
                            "OR ParentBegDate <= {} AND  ParentEndDate >= {})".format(sys_id, today_str,
                                                                                      today_str, today_str))
        return self.cursor.fetchall()

    def get_rows(self, sys_id: int, date: str) -> []:
        """
        Parsing rows from DB and returning list of unique tuples with etalons and list of tuples with data for answers
        """
        rows = []
        data_from_db = self.fetch_from_db(sys_id, date)
        for row in data_from_db:
            try:
                parent_pub_list = [int(pb) for pb in row["ParentPubList"].split(",") if pb != '']
                rows.append(ROW(row["SysID"], row["ID"], row["Cluster"], row["ParentModuleID"],
                                row["ParentID"], parent_pub_list, row["ChildBlockModuleID"],
                                row["ChildBlockID"], row["ModuleID"], row["Topic"], row["Subtopic"],
                                row["DocName"], row["ShortAnswerText"]))
            except ValueError as err:
                pass
        return rows


class ElasticClient(AsyncElasticsearch):
    """Handling with AsyncElasticsearch"""

    def __init__(self, *args, **kwargs):
        self.settings = ElasticSettings()
        self.loop = asyncio.new_event_loop()
        super().__init__(hosts=self.settings.hosts, basic_auth=self.settings.basic_auth,
                         request_timeout=100, max_retries=10, retry_on_timeout=True, *args, **kwargs)

    def texts_search(self, index: str, searching_field: str, texts: [str]) -> []:
        """
        :param searching_field:
        :param texts:
        :param index:
        :param text:
        :return:
        """

        async def search(tx: str, field: str):
            """
            :param field:
            :param tx:
            :return:
            """
            resp = await self.search(
                allow_partial_search_results=True,
                min_score=0,
                index=index,
                # query={"match_phrase": {field: tx}},
                query={"match": {field: tx}},
                size=self.settings.max_hits)
            await self.close()
            return resp

        async def search_texts(s_texts, s_field: str):
            """
            :param s_field: 
            :param s_texts:
            :return:
            """
            texts_search_result = []
            for txt in s_texts:
                res = await search(txt, s_field)
                if [res["hits"]["hits"]]:
                    texts_search_result.append({"text": txt,
                                                "search_results": [{**d["_source"], **{"id": d["_id"]},
                                                                    **{"score": d["_score"]}} for d
                                                                   in res["hits"]["hits"]]})
            return texts_search_result

        return self.loop.run_until_complete(search_texts(texts, searching_field))

    def answer_search(self, index: str, fa_id: int, pub_id: int):
        """отдельный метод для точного поиска по двум полям"""

        async def fa_search(templateId: int, pubId: int):
            """
            :param field:
            :param tx:
            :return:
            """
            resp = await self.search(
                allow_partial_search_results=True,
                min_score=0,
                index=index,
                query={"bool": {"must": [{"match_phrase": {"templateId": templateId}},
                                         {"match_phrase": {"pubId": pubId}}]}},
                size=self.settings.max_hits)
            await self.close()
            return resp

        async def fast_answ_search(faid: int, pbid: int):
            """
            :param s_field:
            :param s_texts:
            :return:
            """
            res = await fa_search(faid, pbid)
            if [res["hits"]["hits"]]:
                return {"search_results": [{**d["_source"], **{"id": d["_id"]},
                                            **{"score": d["_score"]}} for d
                                           in res["hits"]["hits"]]}

        return self.loop.run_until_complete(fast_answ_search(fa_id, pub_id))

    async def search_by_field_exactly(self, index: str, field_name: str, searched_value):
        """
        """
        resp = await self.search(
            allow_partial_search_results=True,
            min_score=0,
            index=index,
            query={"match_phrase": {field_name: searched_value}},
            size=self.settings.max_hits)
        await self.close()
        return resp

    def create_index(self, index_name: str = None) -> None:
        """
        :param index_name:
        """

        async def create(index: str = None) -> None:
            """Creates the index if one does not exist."""
            try:
                await self.indices.create(index=index)
                await self.close()
            except:
                await self.close()
                logger.info("impossible create index with name {}".format(index_name))

        self.loop.run_until_complete(create(index_name))

    def delete_index(self, index_name) -> None:
        """Deletes the index if one exists."""

        async def delete(index: str):
            """
            :param index:
            """
            try:
                await self.indices.delete(index=index)
                await self.close()
            except:
                await self.close()
                logger.info("impossible delete index with name {}".format(index_name))

        self.loop.run_until_complete(delete(index_name))

    async def delete_by_ids(self, index_name: str, del_ids: []):
        """
        :param index_name:
        :param del_ids:
        """
        _gen = ({'_op_type': 'delete',
                 '_index': index_name,
                 '_id': i} for i in del_ids)
        await async_bulk(self, _gen, chunk_size=self.settings.chunk_size, raise_on_error=False, stats_only=True)
        await self.close()

    
    def delete_in_field(self, index: str, field: str, values: []):
        """
        удаление по точному совпадению со значениями в указанном поле
        """

        async def get_ids(index, field, values):
            ids = []
            for value in values:
                res = await self.search_by_field_exactly(index, field, value)
                ids += [d["_id"] for d in res["hits"]["hits"]]
            return ids

        ids_for_del = self.loop.run_until_complete(get_ids(index, field, values))
        print("ids_for_del:", ids_for_del)
        self.loop.run_until_complete(self.delete_by_ids(index, ids_for_del))

    def add_docs(self, index_name: str, docs: [{}]):
        """
        :param index_name:
        :param docs:
        """

        async def add(index: str, dcs: [{}]):
            """Adds documents to the index."""
            _gen = (
                {
                    "_index": index,
                    "_source": doc
                } for doc in dcs
            )
            await async_bulk(self, _gen, chunk_size=self.settings.chunk_size, stats_only=True)
            # await self.close()

        try:
            self.loop.run_until_complete(add(index_name, docs))
            logger.info("adding {} documents in index {}".format(len(docs), index_name))
        except Exception:
            logger.exception("Impossible adding {} documents in index {}".format(len(docs), index_name))
