import { useCallback, useEffect, useState } from "react";
import { deleteDocument, history } from "../api.js";

/** Shared per-user document history — null while loading, [] once loaded. */
export default function useHistoryList() {
  const [items, setItems] = useState(null);

  const refresh = useCallback(() => {
    history().then(setItems).catch(() => setItems([]));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const remove = useCallback(
    async (id) => {
      await deleteDocument(id);
      refresh();
    },
    [refresh]
  );

  return { items, refresh, remove };
}
