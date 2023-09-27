import 'dart:ffi';
import 'package:ffi/ffi.dart';

typedef GeneratorC = Pointer<Utf8> Function(Pointer<Utf8>);
typedef GeneratorDart = Pointer<Utf8> Function(Pointer<Utf8>);

String generateString(String input) {
  final myLib = DynamicLibrary.open('library/lib.so');
  final generator = myLib.lookupFunction<GeneratorC, GeneratorDart>('generator');

  final inputText = input.toNativeUtf8();
  final resultPointer = generator(inputText);
  final result = resultPointer.toDartString();
  

  calloc.free(inputText);
  calloc.free(resultPointer);

  return result;
}