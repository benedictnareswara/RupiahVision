// App shell for M0. Camera, inference and speech arrive in M4; until then the page
// only says plainly that recognition is not available yet.
export default function App() {
  return (
    <main className="app">
      <h1>RupiahVision</h1>
      <p role="status" aria-live="polite" className="status">
        Pengenalan uang belum tersedia. Aplikasi masih dalam pengembangan.
      </p>
    </main>
  );
}
