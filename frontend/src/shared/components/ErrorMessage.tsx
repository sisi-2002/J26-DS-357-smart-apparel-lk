export default function ErrorMessage({ message }: { message: string }) {
  return <div className="error-message" role="alert"><strong>Prediction unavailable</strong><p>{message}</p></div>;
}
